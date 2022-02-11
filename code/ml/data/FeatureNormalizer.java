package ml.data;

import java.util.ArrayList;
import java.util.Collections;

public class FeatureNormalizer implements DataPreprocessor {
	ArrayList<Double> featureMeans;
	ArrayList<Double> featureSDs;
	DataSet train, test;

	public FeatureNormalizer() {
	}

	private void centerData() {

	}

	@Override
	public void preprocessTrain(DataSet train) {
		this.train = train;
		int numFeatures = train.getData().get(0).getFeatureSet().size();

		// MEAN CENTERING
		// sum feature values for each feature across all examples
		featureMeans = new ArrayList<Double>(Collections.nCopies(numFeatures, 0.0));
		for (Example example : train.getData()) {
			ArrayList<Integer> features = new ArrayList<Integer>(example.getFeatureSet());
			for (int i = 0; i < features.size(); i++) {
				featureMeans.set(i, featureMeans.get(i) + example.getFeature(i));
			}
		}

		// divide each feature by size of data set to get mean
		for (int i = 0; i < featureMeans.size(); i++) {
			featureMeans.set(i, featureMeans.get(i) / train.getData().size());
		}

		// subtract mean of each feature from every example's feature value
		for (int i = 0; i < featureMeans.size(); i++) {
			for (Example example : train.getData()) {
				example.setFeature(i, example.getFeature(i) - featureMeans.get(i));
			}
		}

		// VARIANCE SCALING
		// calculate variance for each feature across all examples
		featureSDs = new ArrayList<Double>(Collections.nCopies(numFeatures, 0.0));
		for (Example example : train.getData()) {
			ArrayList<Integer> features = new ArrayList<Integer>(example.getFeatureSet());
			for (int i = 0; i < features.size(); i++) {
				featureSDs.set(i, featureSDs.get(i) + (example.getFeature(i)
						- featureMeans.get(i) * (example.getFeature(i) - featureMeans.get(i))));
			}
		}

		// take square root of variances to get standard deviations
		for (int i = 0; i < train.getData().size(); i++) {
			for (int j = 0; j < train.getData().get(i).getFeatureSet().size(); j++) {
				train.getData().get(i).setFeature(j, train.getData().get(i).getFeature(j)
						/ (Math.sqrt(featureSDs.get(j) / (train.getData().size() - 1))));
			}
		}

//		for (int i = 0; i < featureMeans.size(); i++) {
//			System.out.println(featureMeans.get(i));
//		}
//
//		for (int i = 0; i < featureMeans.size(); i++) {
//			System.out.println(featureSDs.get(i));
//		}

		// check if mean is 0 for all features
//		for (int i = 0; i < train.getAllFeatureIndices().size(); i++) {
//			double m = 0;
//			for (int j = 0; j < train.getData().size(); j++) {
//				m += train.getData().get(j).getFeature(i);
//			}
//			System.out.println("mean: " + m / train.getData().size());
//		}
	}

	@Override
	public void preprocessTest(DataSet test) {
		this.test = test;

		for (Example example : test.getData()) {

			for (int i = 0; i < featureMeans.size(); i++) {
				example.setFeature(i, example.getFeature(i) - featureMeans.get(i));
			}

			for (int i = 0; i < featureSDs.size(); i++) {
				example.setFeature(i,
						example.getFeature(i) / (Math.sqrt(featureSDs.get(i) / (train.getData().size() - 1))));
			}
		}

	}
}