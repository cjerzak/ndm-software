# TB Prior Evaluation Packet

## Request

Evaluate whether the structured TB priors in `ndm` seem reasonable. Focus on
the default prior values, initialized disease-state distributions, and early
forward trajectories across all TB prior specifications.

## Local Forward-Pass Setup

- Repo: `/Users/cjerzak/Documents/ndm-software`
- Runtime: R package loaded from source with JAX CPU backend in conda env
  `jax_cpu`.
- Script: `/Users/cjerzak/Documents/ndm-software/Tmp/tb_prior_forward_sweep.R`
- Output directory:
  `/Users/cjerzak/Documents/ndm-software/Tmp/tb_prior_forward_sweep`
- Built each TB NeuralODE spec without code edits.
- Ran initialized, untrained forward predictions using `testWithoutSampling =
  TRUE` for a deterministic location pass.
- The script also attempted two stochastic seeded passes, but because the JIT
  path had already compiled under `testWithoutSampling = TRUE`, those outputs
  matched the deterministic pass. Treat the stochastic columns as repeated
  deterministic initialized checks, not independent stochastic prior predictive
  draws.

## Cases

Attempted 16 cases: `tb_a`, `tb_b` with `n = 2, 3, 5`, `tb_c`, `tb_d`,
`tb_e`, `tb_f`, `tb_g`, `tb_h`, `tb_i`, `tb_j` with `n = 2, 3, 5`, `tb_k`,
and `tb_l`.

`tb_d` failed in the untrained deterministic forward sweep with:

`ValueError: axis 0 is out of bounds for array of dimension 0`

A separate targeted check using the existing training harness for `tb_d`
completed one SGD step and produced finite predictions:

`loss = 16.15911`, `grad_norm = 1.197466`, `pred_finite = TRUE`.

So interpret the `tb_d` sweep failure as an issue with the untrained/no-sampling
initialized path rather than proof that the trained `tb_d` runtime is unusable.

## Prior Definitions Observed

The structured TB parameter priors are mostly positive rates with
`InvSoftPlus(prior_mean, prior_sd = 0.25)`:

- `lambda = 0.08`
- progression/reactivation rates often `c = 0.05`, `d = 0.14`,
  `e = 0.06`, `f = 0.04`, `k = 0.18`, chain `f_i = 0.12`
- direct/proportion terms use `InvSigmoid`, e.g. `a = 0.20` or `0.25`,
  `b = 0.35`
- time-varying cases use `x1 = 0.06`; `tb_k` has `x2 = 0.15`; `tb_l`
  has `x2 = 0.30`, `x3 = 0.08`

All structured TB init-state TeX priors are `Identity Normal(0, 0.1)`, but the
runtime also synthesizes `InitStateLogit_*` parameters and maps initial states
through a softmax. This distinction matters when judging whether the written
init priors are actually the priors controlling initialized disease-state
proportions.

## Initialized Forward Summary

For successful deterministic initialized passes:

- state values were finite and nonnegative;
- total disease-state mass drift over the short forward horizon was tiny
  (`max_rel_mass_drift <= 6.94e-7`);
- initialized infectious proportion ranged from `2.19e-6` (`tb_j_n5`) to
  `2.21e-2` (`tb_e`);
- mean predicted observation ranged from `3.92e-6` (`tb_j_n5`) to `2.21e-2`
  (`tb_e`);
- time-varying `c_t` stayed finite and positive:
  - `tb_k`: mean `0.0605`, range `0.0584` to `0.0647`
  - `tb_l`: mean `0.0714`, range `0.0672` to `0.0757`

Key deterministic initialized infectious proportions:

| label | n_states | i_init_prop | i_final_prop | pred_mu_mean |
|---|---:|---:|---:|---:|
| tb_j_n5 | 7 | 2.19e-6 | 1.45e-6 | 3.92e-6 |
| tb_j_n3 | 5 | 1.65e-5 | 1.70e-5 | 5.92e-4 |
| tb_j_n2 | 4 | 3.57e-4 | 5.96e-4 | 5.14e-4 |
| tb_h | 4 | 2.90e-3 | 3.02e-3 | 2.97e-3 |
| tb_k | 3 | 4.12e-3 | 4.35e-3 | 4.05e-3 |
| tb_l | 3 | 4.68e-3 | 4.86e-3 | 4.61e-3 |
| tb_f | 4 | 4.72e-3 | 4.84e-3 | 4.79e-3 |
| tb_b_n2 | 5 | 6.74e-3 | 6.74e-3 | 7.56e-3 |
| tb_b_n5 | 8 | 6.77e-3 | 6.77e-3 | 6.79e-3 |
| tb_b_n3 | 6 | 6.92e-3 | 6.95e-3 | 6.88e-3 |
| tb_i | 4 | 7.61e-3 | 7.73e-3 | 7.67e-3 |
| tb_c | 4 | 7.70e-3 | 7.82e-3 | 7.77e-3 |
| tb_a | 3 | 8.85e-3 | 9.11e-3 | 8.79e-3 |
| tb_g | 4 | 2.06e-2 | 2.09e-2 | 2.08e-2 |
| tb_e | 3 | 2.21e-2 | 2.23e-2 | 2.21e-2 |

## Files To Inspect

- `/Users/cjerzak/Documents/ndm-software/R/specs_structured.R`
- `/Users/cjerzak/Documents/ndm-software/tools/runtime_source/ndm_runtime/ModelDefiners/SuperLModel_ParseDynamicODE.R`
- `/Users/cjerzak/Documents/ndm-software/tools/runtime_source/ndm_runtime/ModelDefiners/SuperLModel_BuildML.R`
- `/Users/cjerzak/Documents/ndm-software/Tmp/tb_prior_forward_sweep/tb_forward_report.md`
- `/Users/cjerzak/Documents/ndm-software/Tmp/tb_prior_forward_sweep/tb_forward_summary.csv`
- `/Users/cjerzak/Documents/ndm-software/Tmp/tb_prior_forward_sweep/tb_state_trajectories_summary.csv`
- `/Users/cjerzak/Documents/ndm-software/Tmp/tb_prior_forward_sweep/tb_parameter_samples_summary.csv`
- `/Users/cjerzak/Documents/ndm-software/Tmp/tb_prior_forward_sweep/tb_prior_definitions.csv`
- `/Users/cjerzak/Documents/ndm-software/Tmp/tb_prior_forward_sweep/tb_time_varying_summary.csv`
- `/Users/cjerzak/Documents/ndm-software/Tmp/tb_prior_forward_sweep/tb_forward_errors.csv`

## Questions For GPT-5 Pro

1. Do the rate/fraction prior centers look epidemiologically and numerically
   reasonable for TB-like structures, given the short-horizon initialized
   forward behavior?
2. Are any structures suspicious because their initialized infectious
   proportion or predicted observation is too high, too low, or inconsistent
   with the model semantics?
3. Does the init-state prior implementation look conceptually coherent, given
   the written `Normal(0, 0.1)` init-state priors and the runtime's synthetic
   softmax-logit initialization?
4. What concrete changes or validation checks would you recommend before
   trusting these priors?
