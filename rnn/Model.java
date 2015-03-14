package rnn;

import java.util.List;

public abstract class  Model {
	protected abstract void train(List<Datum> trainData, int _mini_batch_size, int _epoch, double _lr);
	protected abstract double test(List<Datum> testData, String storePath);
	
}
