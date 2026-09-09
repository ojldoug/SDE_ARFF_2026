# Historical-source outcome and modern ablation definition

The historical source search does not uniquely recover Owen's originating split-MLP execution/checkpoint chain. The primary manuscript (`reference/sde-identification/Latest_manuscrip_draft.pdf`, §6.1, Tables2/3 and Appendix Table11), surviving `GPU/lib/lib_Adam_tanh.py`, plotting notebooks and available history provide the following evidence. The detailed prior audit is `results/final_reproduction/HISTORICAL_SPLIT_MLP_AUDIT.md`; the source-search extent is documented in `metric_protocol_reconciliation.md`.

| Setting | Historical evidence / unresolved point |
|---|---|
| Width / architecture | Tanh shallow and two-hidden-layer deep models; Table2 K512 versus Appendix K1024 conflict. Deep caption says K/2 per hidden layer. Exact executed width unproven. |
| Objectives | Sequential regression described, originating split objectives absent; cannot infer covariance loss from the surviving joint trainer. |
| Initialization | Surviving joint source: Glorot drift, zero drift biases; tiny uniform covariance weights/biases. Not proof of split initialization. |
| Epochs / LR / batch | Appendix reports2000 epochs, lr.001, batch512; allocation per split stage and exact table-run association unknown. |
| Covariance | Joint historical source uses symmetric sigma=LL^T, Sigma=(LL^T)^2; missing split implementation prevents transfer of that claim. |
| Checkpoint / cross-fitting / seeds | Exact split checkpoint chain, OOF status and individual seeds not recovered. Manuscript reports30 repetitions. |
| Metrics | Shallow split validation NLL-8.263, drift.8544, Diff..2494; deep split-8.931,.8544,.0701. Diff. semantics and exact evaluation population are not established for the missing trainer. |

None of these aggregates selects the modern configuration. The newly authorized comparison uses the already accepted width27 Joint model and modern split factor pipeline. It must be labeled a modern controlled ablation, not a historical replication.

Both networks have identical final architecture and parameter count (drift893, factor921, total1814), initial arrays per seed, Adam hyperparameters, data and split. Final-model initial weights use the shared Glorot family and zero biases; shared constructor output scaling is unchanged. Covariance is Sigma=LL^T with softplus-positive diagonal of L. Joint optimizes full Gaussian NLL; Split optimizes drift MSE and frozen-residual Gaussian NLL with5-fold OOF targets. Their shared selection rule is earliest minimum canonical-validation stage objective, not identical objective values. Test enters final evaluation only. The additional nuisance fits advance the random stream and add compute: seven300-epoch regressions versus one300-epoch joint fit. These are necessary consequences of the approved two-stage procedure, not tuned differences.
