package rnn;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Collections;
import java.util.List;

public class runtest {
	public static void main(String[] args){
		/*PrintStream out;
		try {
			out = new PrintStream(new FileOutputStream("output.txt"));
			System.setOut(out);
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}*/
		
		
		//1. prepare data
		
		FeatureFactory l = new FeatureFactory();
		List<Datum> trainD=null,testD=null,validD=null;
		double per = 0.85;
		try {
			l.loaddata("/Users/qam/programming/java/rnn/data/wordvector");
			List<Datum> tmp = l.readData("/Users/qam/programming/java/rnn/data/rt-polarity.neg", 1);
			tmp.addAll(l.readData("/Users/qam/programming/java/rnn/data/rt-polarity.pos", 2));
			Collections.shuffle(tmp);
			
			trainD = tmp.subList(0, (int)(tmp.size()*per));
			testD =  tmp.subList((int)(tmp.size()*per), tmp.size());
			System.out.println("Total:"+tmp.size()+" Train Size:"+trainD.size()+" Test Size:"+testD.size());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		//2. use rnn to train data
		RNNModel rnn = new RNNModel(l, 2);
		int iteration = 4;
		for(int i=0; i<iteration; i++){
			System.out.format("\nepoch{%d} \n",i);
			rnn.train(trainD, 15, 20, 0.01);
			rnn.test(testD, null);
		}
		
		//3. predict data and output error rate
		//rnn.test(testD, null);

	}
}
