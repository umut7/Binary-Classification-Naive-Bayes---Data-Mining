import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class main {

    private static final String FILE_PATH = "data.csv";
    private static final int NUM_FOLDS = 10;

    public static void main(String[] args) {
        List<List<String>> data = readCSV(FILE_PATH);
        EvaluationResult evaluationResult = crossValidation(data);
        System.out.println("Micro Averages:");
        printMetrics(evaluationResult.microAvgMetrics);
        System.out.println("\nMacro Averages:");
        printMetrics(evaluationResult.macroAvgMetrics);
    }

    private static List<List<String>> readCSV(String filePath) {
        List<List<String>> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                data.add(Arrays.asList(values));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    private static EvaluationResult crossValidation(List<List<String>> data) {
        int foldSize = data.size() / NUM_FOLDS;
        List<Double> accuracyList = new ArrayList<>();
        List<Double> truePositiveRateList = new ArrayList<>();
        List<Double> trueNegativeRateList = new ArrayList<>();
        List<Double> precisionList = new ArrayList<>();
        List<Double> fScoreList = new ArrayList<>();

        for (int i = 0; i < NUM_FOLDS; i++) {
            List<List<String>> trainData = new ArrayList<>(data);
            List<List<String>> testData = new ArrayList<>();
            for (int j = 0; j < foldSize; j++) {
                testData.add(trainData.remove(i * foldSize));
            }

            NaiveBayesClassifier nbClassifier = new NaiveBayesClassifier(trainData);
            EvaluationMetrics metrics = nbClassifier.evaluate(testData);
            accuracyList.add(metrics.accuracy);
            truePositiveRateList.add(metrics.truePositiveRate);
            trueNegativeRateList.add(metrics.trueNegativeRate);
            precisionList.add(metrics.precision);
            fScoreList.add(metrics.fScore);

            System.out.println("Fold " + (i + 1) + " Accuracy: " + metrics.accuracy);
        }

        EvaluationMetrics microAvgMetrics = calculateMicroAvg(accuracyList, truePositiveRateList,
                trueNegativeRateList, precisionList, fScoreList);
        EvaluationMetrics macroAvgMetrics = calculateMacroAvg(accuracyList, truePositiveRateList,
                trueNegativeRateList, precisionList, fScoreList);

        return new EvaluationResult(microAvgMetrics, macroAvgMetrics);
    }

    private static EvaluationMetrics calculateMicroAvg(List<Double> accuracyList, List<Double> truePositiveRateList, List<Double> trueNegativeRateList, List<Double> precisionList, List<Double> fScoreList) {
        double microAvgAccuracy = calculateAverage(accuracyList);
        double microAvgTruePositiveRate = calculateAverage(truePositiveRateList);
        double microAvgTrueNegativeRate = calculateAverage(trueNegativeRateList);
        double microAvgPrecision = calculateAverage(precisionList);
        double microAvgFScore = calculateAverage(fScoreList);

        return new EvaluationMetrics(microAvgAccuracy, microAvgTruePositiveRate, microAvgTrueNegativeRate,
                microAvgPrecision, microAvgFScore);
    }

    private static EvaluationMetrics calculateMacroAvg(List<Double> accuracyList, List<Double> truePositiveRateList, List<Double> trueNegativeRateList, List<Double> precisionList, List<Double> fScoreList) {
        double macroAvgAccuracy = calculateAverage(accuracyList);
        double macroAvgTruePositiveRate = calculateAverage(truePositiveRateList);
        double macroAvgTrueNegativeRate = calculateAverage(trueNegativeRateList);
        double macroAvgPrecision = calculateAverage(precisionList);
        double macroAvgFScore = calculateAverage(fScoreList);

        return new EvaluationMetrics(macroAvgAccuracy, macroAvgTruePositiveRate, macroAvgTrueNegativeRate,
                macroAvgPrecision, macroAvgFScore);
    }

    private static double calculateAverage(List<Double> values) {

        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.size();
    }

    private static void printMetrics(EvaluationMetrics metrics) {
        System.out.println("Accuracy: " + metrics.accuracy);
        System.out.println("True Positive Rate (Recall): " + metrics.truePositiveRate);
        System.out.println("True Negative Rate: " + metrics.trueNegativeRate);
        System.out.println("Precision: " + metrics.precision);
        System.out.println("F-Score: " + metrics.fScore);
    }
}

class NaiveBayesClassifier {

    private List<List<String>> data;
    private List<String> selectedFeatures;
    private Map<String, Map<String, Integer>> featureCountsByClass;
    private Map<String, Integer> classCounts;

    public NaiveBayesClassifier(List<List<String>> data) {
        this.data = data;
        this.featureCountsByClass = new HashMap<>();
        this.classCounts = new HashMap<>();

        this.selectedFeatures = selectFeatures(data);

        train();
    }

    private List<String> selectFeatures(List<List<String>> data) {
        return data.get(0).subList(0, 5);
    }

    private void train() {
        for (List<String> instance : data) {
            String label = instance.get(instance.size() - 1);

            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);

            for (int i = 0; i < selectedFeatures.size(); i++) {
                String feature = selectedFeatures.get(i);
                String value = instance.get(i);
                featureCountsByClass.putIfAbsent(label, new HashMap<>());
                featureCountsByClass.get(label).put(feature + "=" + value,
                        featureCountsByClass.get(label).getOrDefault(feature + "=" + value, 0) + 1);
            }
        }
    }

    public EvaluationMetrics evaluate(List<List<String>> testData) {
        int truePositive = 0;
        int trueNegative = 0;
        int falsePositive = 0;
        int falseNegative = 0;

        for (List<String> instance : testData) {
            String actualClass = instance.get(instance.size() - 1);
            String predictedClass = classify(instance);
            if (actualClass.equals("A")) {
                if (predictedClass.equals("A")) {
                    truePositive++;
                } else {
                    falseNegative++;
                }
            } else {
                if (predictedClass.equals("A")) {
                    falsePositive++;
                } else {
                    trueNegative++;
                }
            }
        }

        double accuracy = (double) (truePositive + trueNegative) / testData.size();
        double truePositiveRate = (double) truePositive / (truePositive + falseNegative);
        double trueNegativeRate = (double) trueNegative / (trueNegative + falsePositive);
        double precision = (double) truePositive / (truePositive + falsePositive);
        double recall = truePositiveRate;
        double fScore = 2 * (precision * recall) / (precision + recall);

        return new EvaluationMetrics(accuracy, truePositiveRate, trueNegativeRate, precision, fScore);
    }

    private String classify(List<String> instance) {

        double maxProbability = Double.MIN_VALUE;
        String predictedClass = "";

        for (String label : classCounts.keySet()) {
            double classProbability = (double) classCounts.get(label) / data.size();
            double instanceProbability = 1.0;
            for (int i = 0; i < selectedFeatures.size(); i++) {
                String feature = selectedFeatures.get(i);
                String value = instance.get(i);
                int count = featureCountsByClass.get(label).getOrDefault(feature + "=" + value, 0);
                instanceProbability *= (count + 1.0) / (classCounts.get(label) + 2.0);
            }
            double posteriorProbability = classProbability * instanceProbability;
            if (posteriorProbability > maxProbability) {
                maxProbability = posteriorProbability;
                predictedClass = label;
            }
        }
        return predictedClass;
    }
}

class EvaluationMetrics {
    double accuracy;
    double truePositiveRate;
    double trueNegativeRate;
    double precision;
    double fScore;

    public EvaluationMetrics(double accuracy, double truePositiveRate, double trueNegativeRate,
                             double precision, double fScore) {
        this.accuracy = accuracy;
        this.truePositiveRate = truePositiveRate;
        this.trueNegativeRate = trueNegativeRate;
        this.precision = precision;
        this.fScore = fScore;
    }
}


