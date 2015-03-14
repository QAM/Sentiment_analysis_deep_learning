package rnn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.ejml.simple.*;

public class RNNModel extends Model{
	private int batch_size, epoch;
	private double lr;
	 
	private SimpleMatrix W, Wlabel;
	private SimpleMatrix b;
	private FeatureFactory ff;
	public RNNModel(FeatureFactory f, int out_size){
		ff = f;
		Random rm = new Random();
		int fanIn = 2*ff.allVecs.numCols();
		//double fanIn = 100;
		W = SimpleMatrix.random(f.allVecs.numRows(),2*f.allVecs.numRows(), -1/Math.sqrt(fanIn), 1/Math.sqrt(fanIn), rm);
		Wlabel = SimpleMatrix.random(out_size,f.allVecs.numRows(), -1/Math.sqrt(fanIn), 1/Math.sqrt(fanIn), rm);
		b = SimpleMatrix.random(f.allVecs.numRows(),1, -1/Math.sqrt(fanIn), 1/Math.sqrt(fanIn), rm);
	}

	
	@Override
	protected void train(List<Datum> trainData, int _mini_batch_size, int _epoch, double _lr){
		batch_size = _mini_batch_size; epoch = _epoch; lr = _lr;
		for(int i=0; i<epoch; ++i){
			System.out.println(Wlabel);
			System.out.println(W.extractMatrix(0, 20, 0, 20));
			System.out.println(b);
			int now = 0, end = 0;
			while(now < trainData.size()){
				end = (now+batch_size)<trainData.size() ? (now+batch_size):trainData.size();
				updateParameter(trainData.subList(now, end));
				now = end;
			}
		}
	}
	
	private void updateParameter(List<Datum> d){
		SimpleMatrix trainW = new SimpleMatrix(W.numRows(), W.numCols()); trainW.set(0);
		SimpleMatrix trainWlabel = new SimpleMatrix(Wlabel.numRows(), Wlabel.numCols()); trainWlabel.set(0);
		SimpleMatrix trainB = new SimpleMatrix(ff.allVecs.numRows(), 1); trainB.set(0);
		for(int i=0; i<d.size(); ++i){
			List<SimpleMatrix> r = backprop(d.get(i));
			trainWlabel = trainWlabel.plus(r.get(0));
			if(r.size()>1){
				trainW = trainW.plus(r.get(1));
				trainB = trainB.plus(r.get(2));
			}
			System.out.println(trainWlabel.get(0, 0)+" "+trainW.get(0, 0)+" "+trainB.get(0, 0));
			
		}
		
		if(d.size()==0) System.exit(1);
		if(W.get(0, 0)==Double.NaN) System.exit(1);
		System.out.println("before "+W.get(0, 0)+" "+Wlabel.get(0, 0));
		
		W = W.minus((trainW.divide(d.size())).scale(lr));
		Wlabel = Wlabel.minus((trainWlabel.divide(d.size())).scale(lr));
		b = b.minus((trainB.divide(d.size())).scale(lr));
	
		System.out.println("after "+W.get(0, 0)+" "+Wlabel.get(0, 0));
		if(W.get(0, 0)>1){
			System.out.println(trainW.get(0, 0));
			System.exit(0);
		}
	}
	
	private List<SimpleMatrix> backprop(Datum d){
		String[] words = d.sentence.split("[\\s,;\\n\\t]+");
		List<List<SimpleMatrix>> rx = new ArrayList<>();
		List<SimpleMatrix> x = mapWords2Vectors(words);
		SimpleMatrix trainWlabel = new SimpleMatrix(Wlabel.numRows(), Wlabel.numCols()); trainWlabel.set(0);
		List<SimpleMatrix> trainWL = new ArrayList<SimpleMatrix>();
		List<SimpleMatrix> trainB = new ArrayList<SimpleMatrix>();
		List<SimpleMatrix> rz = new ArrayList<SimpleMatrix>();
		List<SimpleMatrix> lossDer = new ArrayList<SimpleMatrix>();
		int y = d.flag;
		rx.add(0,x);
		
		//feedforward
		//at the end, we need to only have one parent
		while(rx.get(0).size()!=1){
			SimpleMatrix ix = rx.get(0).get(0);
			ix = ix.combine(ix.numRows(), 0, rx.get(0).get(1));
			SimpleMatrix z = (W.mult(ix)).plus(b); rz.add(0, z);
			List<SimpleMatrix> ox = new ArrayList<SimpleMatrix>(rx.get(0));
			ox.remove(0); ox.remove(0); ox.add(0, tanh(z));
			rx.add(0,ox);
		}
		SimpleMatrix yhat = softmax(Wlabel.mult(rx.get(0).get(0)));
		
		//TODO backward
		SimpleMatrix trainLoss = softmaxDer(yhat, y);
		lossDer.add(trainLoss);
		if(rz.size()>1){
			lossDer.add( (Wlabel.transpose().mult(lossDer.get(0))).elementMult(tanhDer(rz.get(0))) );
		}
		for(int i=1; i<rz.size(); i++){
			{
				SimpleMatrix tt = ((W.transpose().mult(lossDer.get(i)))
						.extractMatrix(0, ff.allVecs.numRows(), 0, 1)).elementMult(tanhDer(rz.get(i)));
				System.out.println(tt.get(0, 0));
				lossDer.add(tt);
			}
		}
		
		//trainWlabel
		trainWlabel = lossDer.get(0).mult(rx.get(0).get(0).transpose());
		trainWL.add(trainWlabel);
		//trainW
		for(int i=1; i<(lossDer.size()-1); i++){
			SimpleMatrix ix = rx.get(i).get(0);
			ix = ix.combine(ix.numRows(), 0, rx.get(i).get(1));
			{
				trainWL.add(lossDer.get(i).mult(ix.transpose()));
				trainB.add(lossDer.get(i));
			}
		}

		SimpleMatrix trainWADD = new SimpleMatrix(W.numRows(), W.numCols()); trainWADD.set(0);
		while(trainWL.size()>1){
			trainWADD = trainWADD.plus(trainWL.get(1));
			trainWL.remove(1);
		}
		trainWL.add(trainWADD);
		
		
		SimpleMatrix trainBADD = new SimpleMatrix(ff.allVecs.numRows(),1); trainBADD.set(0);
		System.out.println("trainBADD:"+trainBADD.get(0, 0));
		for(int i=0; i<trainB.size(); ++i)
			trainBADD = trainBADD.plus(trainB.get(i));
		trainWL.add(trainBADD);
		
		if(trainWL.get(1).get(0, 0)>1){
			System.out.println("W:"+W);
			System.out.println("trainWL.get(1):"+trainWL.get(1));
		}
		
		return trainWL;
	}

	
	@Override
	protected double test(List<Datum> testData, String storePath) {
		int right = 0;
		for(int i=0; i<testData.size();++i){
			int pred = predict(testData.get(i));
			System.out.println(testData.get(i).flag+" predict:"+pred);
			if(pred == testData.get(i).flag) right++;
		}
		System.out.println(right+"/"+testData.size());
		return 0;
	}
	
	private int predict(Datum d){
		String[] words = d.sentence.split("[\\s,;\\n\\t]+");
		List<List<SimpleMatrix>> rx = new ArrayList<>();
		List<SimpleMatrix> x = mapWords2Vectors(words);
		rx.add(0,x);
		while(rx.get(0).size()!=1){
			SimpleMatrix ix = rx.get(0).get(0);
			ix = ix.combine(ix.numRows(), 0, rx.get(0).get(1));
			SimpleMatrix z = W.mult(ix);
			List<SimpleMatrix> ox = new ArrayList<SimpleMatrix>(rx.get(0));
			ox.remove(0); ox.remove(0); ox.add(0, sigmoid(z));
			rx.add(0,ox);
		}
		SimpleMatrix yhat = softmax(Wlabel.mult(rx.get(0).get(0)));	
		double max = yhat.get(0);
		int index = 1;
		for(int i=1; i<yhat.numRows();++i){
			if(max < yhat.get(i)){
				max = yhat.get(i);
				index = i+1;
			}
		}
		System.out.println(yhat.get(0)+" "+yhat.get(1));
		return index;
	}
	
	private List<SimpleMatrix> mapWords2Vectors(String[] keys){
		List<SimpleMatrix> r = new ArrayList<SimpleMatrix>();
		for(int i=0; i<keys.length; ++i){
			if( ff.wordToNum.get(keys[i]) != null ){
				r.add(ff.allVecs.extractMatrix
						(0, ff.allVecs.numRows(), 
						ff.wordToNum.get(keys[i]), ff.wordToNum.get(keys[i])+1));
			}else{
				SimpleMatrix tmp = new SimpleMatrix(ff.allVecs.numRows(), 1);
				tmp.set(0);
				r.add(tmp);
			}
		}
		return r;
	}

	
	public SimpleMatrix softmax(SimpleMatrix in){
		SimpleMatrix out = new SimpleMatrix(in.numRows(),in.numCols());
		double total = 0;
		for(int j = 0; j < in.numCols(); j++)
			for(int i = 0; i < in.numRows(); i++){
				double t = Math.exp(in.get(i,j));
				total+=t;
				out.set(i,j,t);
			}
		return out.divide(total);
	}
	
	public double softmaxLoss(SimpleMatrix in, int y){
		return Math.log(in.get(y-1,0));
	}
	
	public SimpleMatrix softmaxDer(SimpleMatrix in, int y){
		SimpleMatrix out = new SimpleMatrix(in.numRows(),in.numCols());
		assert(in.numCols()==1);
		for(int i = 0; i < in.numRows(); i++){
			if(i==(y-1)) out.set(i,0,in.get(i,0)-1);
			else out.set(i,0,in.get(i,0));
		}
		return out;
	}
	
	
	
	//api from stanford function
	/**
	 * Performs element-wise tanh function. 
	 */
	public SimpleMatrix tanh(SimpleMatrix in){
		SimpleMatrix out = new SimpleMatrix(in.numRows(),in.numCols());
		for(int j = 0; j < in.numCols(); j++)
			for(int i = 0; i < in.numRows(); i++)
				out.set(i,j,Math.tanh(in.get(i,j)));
		return out;
	}	

	/**
	 * Performs derivative function. 
	 */
	public SimpleMatrix tanhDer(SimpleMatrix in){
		SimpleMatrix out = new SimpleMatrix(in.numRows(),in.numCols());
		out.set(1);
		out.set(out.minus(in.elementMult(in)));
		return out;
	}	

	/**
	 * Performs element-wise sigmoid function.
	 */
	public SimpleMatrix sigmoid(SimpleMatrix in){
		SimpleMatrix out = new SimpleMatrix(in.numRows(),in.numCols());
		for(int j = 0; j < in.numCols(); j++)
			for(int i = 0; i < in.numRows(); i++)
				out.set(i,j,sigmoid(in.get(i,j)));
		return out;
	}	

	/**
	 * Performs element-wise sigmoid function.
	 */
	public SimpleMatrix sigmoidDer(SimpleMatrix in){
		SimpleMatrix ones = new SimpleMatrix(in.numRows(),in.numCols());
		ones.set(1);
		return in.elementMult(ones.minus(in));
	}		


	public static double sigmoid(double x) {
		return (1 / (1 + Math.exp(-x)));
	}

	/**
	 * Performs element-wise tanh function. Fills the new array with these values.
	 */
	public static void elemTanh(SimpleMatrix in, SimpleMatrix out){
		for(int j = 0; j < in.numCols(); j++)
			for(int i = 0; i < in.numRows(); i++)
				out.set(i,j,Math.tanh(in.get(i,j)));
	}



}
