# Aggregate within experiment, replicate, run and checkpoint before pairing.
{
  validated_results <- utils::getFromNamespace(".ndm_report_validate", "ndm")(
    res_mat, training_horizon = get0("report_training_horizon", inherits = TRUE, ifnotfound = NULL)
  )
  res_mat_agg <- utils::getFromNamespace(".ndm_report_aggregate_real", "ndm")(validated_results)
  res_mat_agg <- utils::getFromNamespace(".ndm_report_summaries", "ndm")(res_mat_agg)
  res_mat_agg_maxIters <- res_mat_agg[res_mat_agg$is_final, , drop = FALSE]
}
