.ndm_compile_math_expr <- function(text, symbols = NULL) {
  fail <- function(detail) stop(sprintf("Invalid ODE expression `%s`: %s", text, detail), call. = FALSE)
  parsed <- tryCatch(parse(text = text, keep.source = FALSE), error = function(e) fail(conditionMessage(e)))
  if (length(parsed) != 1L) fail("expected exactly one expression")
  compile <- function(node) {
    if (is.numeric(node) && length(node) == 1L && is.finite(node)) {
      return(format(node, digits = 17L, scientific = TRUE, trim = TRUE))
    }
    if (is.symbol(node)) {
      name <- as.character(node)
      if (!is.null(symbols) && !name %in% symbols) fail(paste("unknown symbol", name))
      return(name)
    }
    if (!is.call(node) || !is.symbol(node[[1L]])) fail("unsupported syntax")
    op <- as.character(node[[1L]])
    args <- as.list(node)[-1L]
    n <- length(args)
    if (!is.null(names(args)) && any(nzchar(names(args)))) fail("named arguments are unsupported")
    if (op == "(" && n == 1L) return(compile(args[[1L]]))
    binary <- c(`+` = "add", `-` = "subtract", `*` = "multiply", `/` = "divide", `^` = "power")
    unary <- c(exp = "exp", sqrt = "sqrt", log = "log")
    if (op %in% c("+", "-") && n == 1L) {
      value <- compile(args[[1L]])
      return(if (op == "+") value else sprintf("jnp$negative(%s)", value))
    }
    if (op %in% names(binary) && n == 2L) {
      return(sprintf("jnp$%s(%s, %s)", binary[[op]], compile(args[[1L]]), compile(args[[2L]])))
    }
    if (op %in% names(unary) && n == 1L) return(sprintf("jnp$%s(%s)", op, compile(args[[1L]])))
    if (op %in% c("SoftPlus", "Sigmoid") && n == 1L) return(sprintf("%s(%s)", op, compile(args[[1L]])))
    if (op %in% c("min", "max") && n >= 1L) {
      compiled <- lapply(args, compile)
      return(Reduce(function(a, b) sprintf("jnp$%s(%s, %s)", if (op == "min") "minimum" else "maximum", a, b), compiled))
    }
    fail(paste("unsupported operator/function or arity:", op))
  }
  compile(parsed[[1L]])
}

# TeX normalization preserves arithmetic groups; neural directives retain their
# existing comma-separated payload and are handled by the neural parser.
.ndm_normalize_math_text <- function(text, neural = FALSE) {
  text <- gsub("\\\\", "", text)
  text <- gsub("cdot", "*", text, fixed = TRUE)
  text <- gsub("_\\{([[:alnum:]_]+)\\}", "_\\1", text)
  if (neural) return(gsub("[{}]", "", text))
  group <- function(s, start) {
    while (start <= nchar(s) && grepl("\\s", substr(s, start, start))) start <- start + 1L
    if (substr(s, start, start) != "{") stop("A TeX fraction requires two braced arguments.", call. = FALSE)
    depth <- 1L; end <- start
    while (depth > 0L && end < nchar(s)) {
      end <- end + 1L
      ch <- substr(s, end, end)
      depth <- depth + as.integer(ch == "{") - as.integer(ch == "}")
    }
    if (depth != 0L) stop("Unbalanced TeX arithmetic group.", call. = FALSE)
    list(text = substr(s, start + 1L, end - 1L), end = end)
  }
  repeat {
    at <- regexpr("frac\\s*\\{", text)[[1L]]
    if (at < 0L) break
    numerator <- group(text, at + 4L)
    denominator <- group(text, numerator$end + 1L)
    replacement <- sprintf("((%s)/(%s))", .ndm_normalize_math_text(numerator$text), .ndm_normalize_math_text(denominator$text))
    text <- paste0(substr(text, 1L, at - 1L), replacement, substr(text, denominator$end + 1L, nchar(text)))
  }
  chartr("{}", "()", text)
}
