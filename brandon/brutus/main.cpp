#include <iostream>
using namespace std;
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric> 
#include <cstdlib>
#include "mpreal.h"
using namespace mpfr;
#include "Brutus.h"
#include <iterator>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>

string tostr(int x);
string tostr(double x);

class CSVReader{
	std::string fileName;
	std::string delimeter;
 
public:
	CSVReader(std::string filename, std::string delm = ",") :
			fileName(filename), delimeter(delm)
	{ }
 
	// Function to fetch data from a CSV File
	std::vector<std::vector<std::string> > getData();
};

std::vector<std::vector<std::string> > CSVReader::getData(){
	std::ifstream file(fileName);
	std::vector<std::vector<std::string> > dataList;
	std::string line = "";

	// Iterate through each line and split the content using delimeter
	while (getline(file, line)){
		std::vector<std::string> vec;
		boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
		dataList.push_back(vec);
	}
	// Close the File
	file.close();
 
	return dataList;
}


int main(int argc, char* argv[]) {

	if(argc != 2){
		cerr << "Enter file no" << endl;
		exit(1);
	}

	int fileNum = atoi(argv[1]);
	int batchNum = 10;
	string filename = "/nBodyData/inputs/indat_"+tostr(batchNum)+"_"+tostr(fileNum)+".dat";

	mpreal t_end = "10.0";
	mpreal eta = "0.24";
	mpreal epsilon = "1e-11"; // = "1e-6"; // Bulirsch-Stoer tolerance
	int numBits = 128;
	double dt = 0.00390625;
	int pmax = 4;

	cout << "\nStarting params: ";
	cout << " tend=" << t_end <<  " dt=" << dt << " eps=" << epsilon  <<  " lw=" << numBits << " pmax=" << pmax <<endl;

	std::ifstream file;
	file.open(filename); 
	if(file.fail()){
		cerr << "Could not open input file: " << filename << endl;
		exit(1);
	}

	vector<vector<string>> events = {};
	CSVReader reader(filename);
	events = reader.getData();
	file.close();

	cout << "Using input: " << filename << endl;

	string fileString = "/nBodyData/brutusSim/batch_brutus"+tostr(batchNum)+"_"+tostr(fileNum)+".csv";
	char fileChar [] = {};
	strcpy(fileChar, fileString.c_str());
	remove(fileChar);
	cout << "Using output: " << fileString << endl;

	fileNum = atoi(argv[1]);

	// std::ofstream fB; 
	// fB.open(fileString, ios_base::app);
	// string headerString = "file,eventID,m1,m2,m3,x1,x2,x3,y1,y2,y3,dx1,dx2,dx3,dy1,dy2,dy3,tEnd,x1tEnd,x2tEnd,x3tEnd,y1tEnd,y2tEnd,y3tEnd,dx1tEnd,dx2tEnd,dx3tEnd,dy1tEnd,dy2tEnd,dy3tEnd,e1,e2,e3";
	// fB << headerString << endl;
	// fB.close();

	cout << "Evolving model... " << std::flush;
	int eventID = 10000;
	int previousEvent = 0;

	for(int iEvent=0; iEvent < events.size(); ++iEvent){

		vector<string> vec = events[iEvent];
		if(vec.size() < 1){
			cerr << "Error: event not found!" << endl;
			continue;
		}
		int n = stoi(vec[vec.size()-1]);
		vec.pop_back();

		mpreal t = "0";
		vector<double> data(7*n);
		vector<mpreal> datmp(7*n);

		if(vec.size() < 7*n){
			cerr << "Input file not read correctly." << endl;
			exit(1);
		}
		if(vec.size() != data.size()){
			cerr << "Input file is incorrect size" << endl;
			exit(1);
		}

		for(int iDatapoint=0; iDatapoint < vec.size(); iDatapoint++){
			data[iDatapoint] = stod(vec[iDatapoint]);
			datmp[iDatapoint] = (mpreal)data[iDatapoint];
		}

		Brutus brutus = Brutus(t, datmp, epsilon, numBits, pmax);
		vector<string> v = brutus.get_data_string();

		/*
		v indices: 
		0: m1    1: p1x  2: p1y    3: p1z   4: p1vx   5: p1vy   6: p1vz
		7: m2    8: p2x  9: p2y   10: p2z  11: p2vx  12: p2vy  13: p2vz
		14: m3  15: p3x  16: p3y  17: p3z  18: p3vx  19: p3vy  20: p3vz
		...

		data indices: same as above
		*/ 

		int count = 0;
		string eventString = "";

		while(t<t_end){
			t+=dt;
 
			if(eventID/10000 != previousEvent){
				cout << previousEvent << std::flush;
				previousEvent = eventID/10000;
			}

			count += 1;
			
			bool converged = brutus.evolve((mpreal)t);
			if(!converged){ break; }

			v = brutus.get_data_string();

			eventString += to_string(fileNum) + "," + tostr(eventID+count) + ",";
			string mass_str(""), xpos(""), ypos(""), zpos(""), vxstr(""), vystr(""), vzstr(""), estr(""), istrx(""), istry(""), istrz(""), istrvx(""), istrvy(""), istrvz("");

			for(int i=0; i < n; ++i){
				mass_str += v[i*7] + ",";

				istrx += to_string(data[(i*7) + 1]) + ",";
				istry += to_string(data[(i*7) + 2]) + ",";
				istrz += to_string(data[(i*7) + 3]) + ",";
				istrvx += to_string(data[(i*7) + 4]) + ",";
				istrvy += to_string(data[(i*7) + 5]) + ",";
				istrvz += to_string(data[(i*7) + 6]) + ",";

				xpos += v[(i*7) + 1] + ",";
				ypos += v[(i*7) + 2] + ",";
				zpos += v[(i*7) + 3] + ",";
				vxstr += v[(i*7) + 4] + ",";
				vystr += v[(i*7) + 5] + ",";
				vzstr += v[(i*7) + 6] + ",";
			}
			string tstr = t.toString() + ",";
			
			eventString += mass_str + istrx + istry + istrz + istrvx + istrvy + istrvz;
			eventString += tstr + xpos + ypos + zpos + vxstr + vystr + vzstr + "\n";
		}

		eventID += 10000;

		// cout << " Writing event to file.." << std::flush;
		std::ofstream fB; 
		fB.open(fileString, ios_base::app);
		fB << eventString << endl;
		fB.close();
		// cout << " Finished writing" << std::flush;
	}

	return 0;
}


string tostr(int x){
	stringstream str;
	str << x;
	return str.str();
}

string tostr(double x){
	stringstream str;
	str << x;
	return str.str();
}