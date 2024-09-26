class EvaluationResult {
    EvaluationMetrics microAvgMetrics;
    EvaluationMetrics macroAvgMetrics;

    public EvaluationResult(EvaluationMetrics microAvgMetrics, EvaluationMetrics macroAvgMetrics) {
        this.microAvgMetrics = microAvgMetrics;
        this.macroAvgMetrics = macroAvgMetrics;
    }
}