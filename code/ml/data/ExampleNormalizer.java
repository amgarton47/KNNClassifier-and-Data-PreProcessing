package ml.data;

import java.util.ArrayList;

/**
 * A class used to pre-process testing and training data. This class does so by
 * changing all example feature values so that the example has length 1.
 * 
 * 
 * @author Aidan Garton
 *
 */
public class ExampleNormalizer implements DataPreprocessor {

	private ArrayList<Double> trainLengths, testLengths;

	public ExampleNormalizer() {
	}

	@Override
	public void preprocessTrain(DataSet train) {
		trainLengths = new ArrayList<Double>(train.getData().size());

		for (Example example : train.getData()) {

			double sumOfFeaturesSquared = 0;

			// calculate and store the sizes of each example
			for (int i = 0; i < example.getFeatureSet().size(); i++) {
				sumOfFeaturesSquared += example.getFeature(i) * example.getFeature(i);
			}

			trainLengths.add(Math.sqrt(sumOfFeaturesSquared));
		}

		for (int j = 0; j < train.getData().size(); j++) {
			Example example = train.getData().get(j);
			// divide each feature of example by its size
			for (int i = 0; i < example.getFeatureSet().size(); i++) {
				example.setFeature(i, example.getFeature(i) / trainLengths.get(j));
			}
		}

		// double check that lengths are 1
//		for (Example example : train.getData()) {
//			double n = 0;
//			for (int i = 0; i < example.getFeatureSet().size(); i++) {
//				n += example.getFeature(i) * example.getFeature(i);
//			}
//
//			System.out.println(Math.sqrt(n));
//		}

	}

	@Override
	public void preprocessTest(DataSet test) {
		testLengths = new ArrayList<Double>();

		for (Example example : test.getData()) {

			double sumOfFeaturesSquared = 0;

			// calculate and store the sizes of each example
			for (int i = 0; i < example.getFeatureSet().size(); i++) {
				sumOfFeaturesSquared += example.getFeature(i) * example.getFeature(i);
			}

			testLengths.add(Math.sqrt(sumOfFeaturesSquared));
		}

		for (int j = 0; j < test.getData().size(); j++) {
			Example example = test.getData().get(j);
			// divide each feature of example by its size
			for (int i = 0; i < example.getFeatureSet().size(); i++) {
				example.setFeature(i, example.getFeature(i) / testLengths.get(j));
			}
		}
	}
}
