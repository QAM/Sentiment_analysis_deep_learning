package rnn;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.ejml.simple.*;
import org.ejml.data.*;
import org.ejml.ops.*;


public class FeatureFactory {
	HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	HashMap<Integer, String> numToWord = new HashMap<Integer, String>();
	SimpleMatrix allVecs;
	
	public FeatureFactory() {

	}
	
	public List<Datum> readData(String Filename) throws IOException{
		return readData(Filename, 0);
	}
	
	public List<Datum> readData(String Filename, int flag) throws IOException{
		ArrayList<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(Filename));
		int counter = 0;
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			data.add(new Datum(line, flag));
		}
		return data;
	} 
	
	public void loaddata(String vocabFilename) throws IOException {
		int dimension = 100;
		int vectorSize = 0;
		ArrayList<String> data = new ArrayList<String>();
		// reading in vocab list
		BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
		int counter = 0;
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			data.add(line);
		}
		
		// reading in matrix
		allVecs = new SimpleMatrix(dimension, data.size());
		counter = 0;
		for (int i=0; i<data.size();++i) {
			String[] bits = data.get(i).split(" ");
			vectorSize = bits.length-1;
			String word = bits[0];
			numToWord.put(counter,word);
			for (int pos=0;pos<vectorSize;pos++){ 
				allVecs.set(pos, counter, Double.parseDouble(bits[pos+1]));		
			}
			counter++;
		}
		assert(counter == wordToNum.size());		
		
		
		
	} 
	
}
