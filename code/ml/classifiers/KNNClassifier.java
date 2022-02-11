package ml.classifiers;

import java.util.ArrayList;
import java.util.PriorityQueue;

import ml.data.DataSet;
import ml.data.Example;
import ml.data.ExampleNormalizer;
import ml.data.FeatureNormalizer;

/**
 * 
 * @author Aidan Garton
 *
 */
public class KNNClassifier implements Classifier {
	private int k = 3;
	private ArrayList<DistanceExample> distances;
	private DataSet data;

	public KNNClassifier() {
	}

	public void setK(int k) {
		this.k = k;
	}

	@Override
	public void train(DataSet data) {
		// do nothing?

		// ok maybe just this?
		this.data = data;
	}

	@Override
	public double classify(Example example) {
		distances = new ArrayList<DistanceExample>(data.getData().size());

		// calculate distance between example and all other examples
		// and store them i distances array
		for (Example e : data.getData()) {
			double d = 0;
			// d = sqrt( (a1-b1)^2 +(a1-b2)^2 + ... + (an-bn)^2 )
			for (int i = 0; i < data.getAllFeatureIndices().size(); i++) {
				d += (e.getFeature(i) - example.getFeature(i)) * (e.getFeature(i) - example.getFeature(i));
			}

			distances.add(new DistanceExample(e, Math.sqrt(d)));
		}

		PriorityQueue<DistanceExample> distancesPQ = new PriorityQueue<DistanceExample>(distances);

		// count majority label of nearest k neighbors
		int c0 = 0, c1 = 0;
		for (int i = 0; i < k; i++) {
			DistanceExample de = distancesPQ.remove();
			if (de.getExample().getLabel() == -1) {
				c0++;
			} else {
				c1++;
			}
		}

//		while (!distancesPQ.isEmpty()) {
//			System.out.println(distancesPQ.remove().getDistance());
//		}

		// return majority label
		if (c0 >= c1) {
			return -1;
		} else {
			return 1;
		}
	}

	public static void main(String[] args) {
		DataSet data = new DataSet("data/titanic-train.real.csv");
		FeatureNormalizer featureNormalizer = new FeatureNormalizer();
		ExampleNormalizer exampleNormalizer = new ExampleNormalizer();

//		System.out.println(data.getData().get(0).getFeature(0));
		featureNormalizer.preprocessTrain(data);
		exampleNormalizer.preprocessTrain(data);
//		System.out.println(data.getData().get(0).getFeature(0));

		KNNClassifier knn = new KNNClassifier();
		knn.train(data);

//		double predict = knn.classify(data.getData().get(0));

		PerceptronClassifier p = new PerceptronClassifier();
		p.train(data);
		AveragePerceptronClassifier ap = new AveragePerceptronClassifier();
		ap.train(data);

		double correct = 0, total = 0;
		for (Example e : data.getData()) {
//			double predict = knn.classify(e);
			double predict = ap.classify(e);
//
//			System.out.println(e.getLabel() + " " + predict);
//
//			System.out.println(e.getFeature(0));
			if (e.getLabel() == predict) {
				correct++;
			}
			total++;
		}

		System.out.println("accuracy: " + correct / total);
	}
}
