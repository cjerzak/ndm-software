pkgname <- "ndm"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('ndm')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("ndm_build_model")
### * ndm_build_model

flush(stderr()); flush(stdout())

### Name: ndm_build_model
### Title: Build and train models with package-managed runtime stages
### Aliases: ndm_build_model print.ndm_model ndm_train
###   print.ndm_trained_model ndm_fit

### ** Examples

cfg <- ndm_create_config()
spec <- ndm_model_spec()
## Not run: 
##D runtime_env <- ndm_prepare_runtime(cfg)
##D ndm_prepare_data(runtime_env, generator = "sim")
##D model <- ndm_build_model(runtime_env, model_spec = spec)
##D trained <- ndm_train(model)
## End(Not run)




cleanEx()
nameEx("ndm_check_backend")
### * ndm_check_backend

flush(stderr()); flush(stdout())

### Name: ndm_check_backend
### Title: Provision and initialize the Python backend
### Aliases: ndm_check_backend ndm_build_backend ndm_initialize_backend
###   ndm_backend_modules

### ** Examples

ndm_check_backend()

## Not run: 
##D ndm_build_backend()
## End(Not run)

## Not run: 
##D backend <- ndm_initialize_backend()
##D backend$default_backend
## End(Not run)

## Not run: 
##D ndm_initialize_backend()
##D ndm_backend_modules()
## End(Not run)




cleanEx()
nameEx("ndm_create_config")
### * ndm_create_config

flush(stderr()); flush(stdout())

### Name: ndm_create_config
### Title: Create runtime configuration objects
### Aliases: ndm_create_config print.ndm_config

### ** Examples

cfg <- ndm_create_config()
print(cfg)




cleanEx()
nameEx("ndm_create_real_run_config")
### * ndm_create_real_run_config

flush(stderr()); flush(stdout())

### Name: ndm_create_real_run_config
### Title: Package-native run configurations and workflow runners
### Aliases: ndm_create_real_run_config ndm_create_sim_run_config
###   ndm_create_multidisease_run_config ndm_run_real ndm_run_sim
###   ndm_run_multidisease

### ** Examples

## Not run: 
##D cfg <- ndm_create_sim_run_config(
##D   project_root = getwd(),
##D   grid_file = "Data/RunGrids/SimGrids/SimGrid_BigSimsLatest.csv",
##D   outer = 1L,
##D   dry_run = TRUE
##D )
##D 
##D ndm_run_sim(cfg)
## End(Not run)



cleanEx()
nameEx("ndm_create_tfrecord_parser")
### * ndm_create_tfrecord_parser

flush(stderr()); flush(stdout())

### Name: ndm_create_tfrecord_parser
### Title: Create and read TensorFlow TFRecord datasets
### Aliases: ndm_create_tfrecord_parser ndm_read_tfrecord_dataset
###   ndm_collect_tfrecord_batches

### ** Examples

fields <- ndm_tfrecord_fields_real()
parser <- ndm_create_tfrecord_parser(fields, ndm_tfrecord_dtype_map("real"))
is.function(parser)




cleanEx()
nameEx("ndm_model_spec_presets")
### * ndm_model_spec_presets

flush(stderr()); flush(stdout())

### Name: ndm_model_spec_presets
### Title: Inspect and create epidemic model specifications
### Aliases: ndm_model_spec_presets ndm_model_spec ndm_model_spec_from_tex
###   ndm_model_spec_to_tex print.ndm_model_spec

### ** Examples

ndm_model_spec_presets()

spec <- ndm_model_spec()
print(spec)

spec <- ndm_model_spec()
tex_path <- tempfile(fileext = ".tex")
ndm_model_spec_to_tex(spec, tex_path)
ndm_model_spec_from_tex(tex_path)

spec <- ndm_model_spec()
tex <- ndm_model_spec_to_tex(spec)
nchar(tex) > 0




cleanEx()
nameEx("ndm_new_runtime_env")
### * ndm_new_runtime_env

flush(stderr()); flush(stdout())

### Name: ndm_new_runtime_env
### Title: Create and populate runtime environments
### Aliases: ndm_new_runtime_env ndm_set_runtime_globals

### ** Examples

env <- ndm_new_runtime_env()
ndm_set_runtime_globals(env, list(example_value = 1L))
env$example_value




cleanEx()
nameEx("ndm_predict")
### * ndm_predict

flush(stderr()); flush(stdout())

### Name: ndm_predict
### Title: Generate predictions or evaluate the training loss
### Aliases: ndm_predict ndm_loss

### ** Examples

## Not run: 
##D preds <- ndm_predict(model, batch = batch_l_cal)
##D loss <- ndm_loss(model, batch = batch_l_cal)
## End(Not run)




cleanEx()
nameEx("ndm_prepare_runtime")
### * ndm_prepare_runtime

flush(stderr()); flush(stdout())

### Name: ndm_prepare_runtime
### Title: Prepare a local runtime for execution
### Aliases: ndm_prepare_runtime ndm_prepare_data

### ** Examples

env <- ndm_new_runtime_env()
cfg <- ndm_create_config()
## Not run: 
##D ndm_prepare_runtime(cfg, env)
## End(Not run)

env <- ndm_new_runtime_env()
ndm_set_runtime_globals(env, list(example_value = 1L))




cleanEx()
nameEx("ndm_print")
### * ndm_print

flush(stderr()); flush(stdout())

### Name: ndm_print
### Title: Print a timestamped diagnostic message
### Aliases: ndm_print

### ** Examples

ndm_print("starting")




cleanEx()
nameEx("ndm_tf_batch_to_r")
### * ndm_tf_batch_to_r

flush(stderr()); flush(stdout())

### Name: ndm_tf_batch_to_r
### Title: Convert and bundle TFRecord batches
### Aliases: ndm_tf_batch_to_r ndm_tf_batch_to_jax
###   ndm_batch_to_model_inputs ndm_load_tfrecord_bundle
###   ndm_attach_tfrecord_bundle

### ** Examples

batch <- list(
  XPred = array(0, c(2, 3, 4)),
  XPred_mask = array(TRUE, c(2, 3, 4)),
  Context = array(0, c(2, 1, 4)),
  Context_mask = array(TRUE, c(2, 1, 4)),
  location_id_numeric = matrix(1L, nrow = 2),
  time_id_numeric = matrix(1L, nrow = 2)
)
packaged <- ndm_batch_to_model_inputs(batch)
length(packaged)




cleanEx()
nameEx("ndm_tfrecord_fields_real")
### * ndm_tfrecord_fields_real

flush(stderr()); flush(stdout())

### Name: ndm_tfrecord_fields_real
### Title: Inspect TFRecord field contracts
### Aliases: ndm_tfrecord_fields_real ndm_tfrecord_fields_sim
###   ndm_tfrecord_dtype_map

### ** Examples

ndm_tfrecord_fields_real()
ndm_tfrecord_dtype_map("sim")




### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
