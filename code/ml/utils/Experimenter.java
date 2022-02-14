package ml.utils;

import java.util.ArrayList;

import ml.classifiers.AveragePerceptronClassifier;
import ml.classifiers.Classifier;
import ml.classifiers.KNNClassifier;
import ml.classifiers.PerceptronClassifier;
import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.data.ExampleNormalizer;
import ml.data.FeatureNormalizer;

public class Experimenter {

	public static double[] performNFoldTest(Classifier classifier, DataSet data, int numSplits, int numTrialsPerSplit) {
		System.out.println("Testing...");
		CrossValidationSet vs = new CrossValidationSet(data, numSplits);
		double[] accuracies = new double[numSplits];

		for (int i = 0; i < numSplits; i++) {
			DataSetSplit split = vs.getValidationSet(i, true);

			double correct = 0, total = 0;
			for (int j = 0; j < numTrialsPerSplit; j++) {
				classifier.train(split.getTrain());

				ArrayList<Example> test = split.getTest().getData();

				for (int k = 0; k < test.size(); k++) {
					if (classifier.classify(test.get(k)) == test.get(k).getLabel()) {
						correct++;
					}
					total++;
				}
			}
			accuracies[i] = correct / total;
		}
		System.out.println("Done.\n");
		return accuracies;
	}

	public static void main(String[] args) {
		DataSet data = new DataSet("data/titanic-train.csv");
		DataSet dataReal = new DataSet("data/titanic-train.real.csv");
//
//		System.out.println("1: AveragePerceptron on \"old\" data.");
//		System.out.println("_____________________________________");
//		AveragePerceptronClassifier ap1 = new AveragePerceptronClassifier();
//		ap1.setIterations(10);
//
//		double[] accuracies1 = performNFoldTest(ap1, data, 10, 100);
//
//		double sum1 = 0;
//		for (int i = 0; i < accuracies1.length; i++) {
//			System.out.println("Split " + i + " accuracy: " + Math.floor(accuracies1[i] * 100000) / 1000 + "\\% \\\\");
//			sum1 += accuracies1[i];
//		}
////		System.out.println("-------------------------");
//		System.out.println("Average accuracy: " + Math.floor((sum1 / accuracies1.length) * 100000) / 1000 + "\\%\n");
//
//		//
//
//		System.out.println("2: AveragePerceptron on \"new\" data.");
//		System.out.println("_____________________________________");
//		AveragePerceptronClassifier ap2 = new AveragePerceptronClassifier();
//		ap2.setIterations(10);
//
//		double[] accuracies2 = performNFoldTest(ap2, dataReal, 10, 100);
//
//		double sum2 = 0;
//		for (int i = 0; i < accuracies2.length; i++) {
//			System.out.println("Split " + i + " accuracy: " + Math.floor(accuracies2[i] * 100000) / 1000 + "\\% \\\\");
//			sum2 += accuracies2[i];
//		}
//		System.out.println("-------------------------");
//		System.out.println("Average accuracy: " + Math.floor((sum2 / accuracies2.length) * 100000) / 1000 + "\\%\n");
//
//		//
//
//		System.out.println("1: KNN on \"old\" data.");
//		System.out.println("_____________________________________");
//		KNNClassifier knn1 = new KNNClassifier();
//
//		double[] accuracies3 = performNFoldTest(knn1, data, 10, 100);
//
//		double sum3 = 0;
//		for (int i = 0; i < accuracies3.length; i++) {
//			System.out.println("Split " + i + " accuracy: " + Math.floor(accuracies3[i] * 100000) / 1000 + "\\% \\\\");
//			sum3 += accuracies3[i];
//		}
//		System.out.println("-------------------------");
//		System.out.println("Average accuracy: " + Math.floor((sum3 / accuracies3.length) * 100000) / 1000 + "\\%\n");
//
//		//
//
//		System.out.println("1: KNN on \"new\" data.");
//		System.out.println("_____________________________________");
//		KNNClassifier knn2 = new KNNClassifier();
//
//		double[] accuracies4 = performNFoldTest(knn2, dataReal, 10, 100);
//
//		double sum4 = 0;
//		for (int i = 0; i < accuracies4.length; i++) {
//			System.out.println("Split " + i + " accuracy: " + Math.floor(accuracies4[i] * 100000) / 1000 + "\\% \\\\");
//			sum4 += accuracies4[i];
//		}
//		System.out.println("-------------------------");
//		System.out.println("Average accuracy: " + Math.floor((sum4 / accuracies4.length) * 100000) / 1000 + "\\%\n");

		//

		double d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0;

		String str = "";
		for (int i = 0; i < 10; i++) {
			str += "fold " + i + " & ";

			CrossValidationSet x = new CrossValidationSet(dataReal, 10);
			DataSetSplit s = x.getValidationSet(i, true);

			ExampleNormalizer en = new ExampleNormalizer();
			en.preprocessTrain(s.getTrain());
			en.preprocessTest(s.getTest());

			KNNClassifier k = new KNNClassifier();
			k.train(s.getTrain());

			double correct = 0, total = 0;
			for (int j = 0; j < 100; j++) {

				for (Example e : s.getTest().getData()) {
					if (e.getLabel() == k.classify(e)) {
						correct++;
					}
					total++;
				}
			}
			d1 += correct / total;
			str += Math.floor(correct / total * 100000) / 1000 + "\\% & ";

			//

			DataSetSplit s1 = x.getValidationSet(i, true);

			FeatureNormalizer fn = new FeatureNormalizer();
			fn.preprocessTrain(s1.getTrain());
			fn.preprocessTest(s1.getTest());

			KNNClassifier k1 = new KNNClassifier();
			k1.train(s1.getTrain());

			double correct1 = 0, total1 = 0;
			for (int j = 0; j < 100; j++) {

				for (Example e : s1.getTest().getData()) {
					if (e.getLabel() == k1.classify(e)) {
						correct1++;
					}
					total1++;
				}
			}
			d2 += correct1 / total1;
			str += Math.floor(correct1 / total1 * 100000) / 1000 + "\\% & ";

			//
			DataSetSplit s2 = x.getValidationSet(i, true);

			FeatureNormalizer fn1 = new FeatureNormalizer();
			fn1.preprocessTrain(s2.getTrain());
			fn1.preprocessTest(s2.getTest());

			ExampleNormalizer en1 = new ExampleNormalizer();
			en1.preprocessTrain(s2.getTrain());
			en1.preprocessTest(s2.getTest());

			KNNClassifier k2 = new KNNClassifier();
			k2.train(s2.getTrain());

			double correct2 = 0, total2 = 0;
			for (int j = 0; j < 100; j++) {

				for (Example e : s2.getTest().getData()) {
					if (e.getLabel() == k2.classify(e)) {
						correct2++;
					}
					total2++;
				}
			}
			d3 += correct2 / total2;
			str += Math.floor(correct2 / total2 * 100000) / 1000 + "\\% & ";

			//

			AveragePerceptronClassifier p = new AveragePerceptronClassifier();
			p.setIterations(10);
			p.train(s.getTrain());

			double correct3 = 0, total3 = 0;
			for (int j = 0; j < 100; j++) {

				for (Example e : s.getTest().getData()) {
					if (e.getLabel() == p.classify(e)) {
						correct3++;
					}
					total3++;
				}
			}
			d4 += correct3 / total3;
			str += Math.floor(correct3 / total3 * 100000) / 1000 + "\\% & ";

			//

			AveragePerceptronClassifier p1 = new AveragePerceptronClassifier();
			p1.setIterations(10);
			p1.train(s1.getTrain());

			double correct4 = 0, total4 = 0;
			for (int j = 0; j < 100; j++) {

				for (Example e : s1.getTest().getData()) {
					if (e.getLabel() == p1.classify(e)) {
						correct4++;
					}
					total4++;
				}
			}
			d5 += correct4 / total4;
			str += Math.floor(correct4 / total4 * 100000) / 1000 + "\\% & ";

//

			AveragePerceptronClassifier p2 = new AveragePerceptronClassifier();
			p2.setIterations(10);
			p2.train(s1.getTrain());

			double correct5 = 0, total5 = 0;
			for (int j = 0; j < 100; j++) {

				for (Example e : s2.getTest().getData()) {
					if (e.getLabel() == p2.classify(e)) {
						correct5++;
					}
					total5++;
				}
			}
			d6 += correct5 / total5;
			str += Math.floor(correct5 / total5 * 100000) / 1000 + "\\% \\\\";

			System.out.println(str);
			str = "";
		}
		d1 /= 10;
		d2 /= 10;
		d3 /= 10;
		d4 /= 10;
		d5 /= 10;
		d6 /= 10;

		d1 = Math.floor(d1 * 100000) / 1000;
		d2 = Math.floor(d2 * 100000) / 1000;
		d3 = Math.floor(d3 * 100000) / 1000;
		d4 = Math.floor(d4 * 100000) / 1000;
		d5 = Math.floor(d5 * 100000) / 1000;
		d6 = Math.floor(d6 * 100000) / 1000;
		
		System.out.println("Average of folds: & " + d1 + "\\% & " + d2 + "\\% & " + d3 + "\\% & " + d4 + "\\% & " + d5 + "\\% & " + d6 + "\\%");
	}
}
