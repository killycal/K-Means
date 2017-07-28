#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <windows.h>
#include <string>

using namespace std;
int N, D,K;
double bestrand=DBL_MAX;
double bestjaccards=DBL_MAX;
struct Point
{
	int grp=-1;
	vector<double> dim;
	double distance=0;
};
vector<Point> master;
vector<Point> tru;
vector<vector<Point>> clusters;
ofstream output;
ofstream output2;
ofstream output3;
string name;
vector<double> SSE;
vector<Point> centroids;
void EV()
{
	int TP = 0;
	int TN = 0; 
	int F = 0;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (i != j)
			{
				if (master[i].grp == master[j].grp)
				{
					if (tru[i].grp == tru[j].grp)
						TP++;
					else
						F++;
				}
				else if (master[i].grp != master[j].grp)
				{
					if (tru[i].grp != tru[j].grp)
						TN++;
					else
						F++;
				}
			}
		}
	}
	double tp = TP;
	double f = F;
	double tn = TN;
	double jaccard = tp / (tp+f);
	double rand= (tp+tn)/N;
	cout << "Jaccard: " << jaccard<< "     Rand: " << rand <<endl;
	if (jaccard < bestjaccards)
		bestjaccards = jaccard;
	if (rand < bestrand)
		bestrand = rand;
}
void groupClusters(int k)
{
	clusters.clear();
	
	for (int i = 0; i < k; i++)//for every cluster
	{
		vector<Point> point;
		clusters.push_back(point);
	}

	for (int i = 0; i < N; i++)//for every point
	{
		for (int j = 0; j < k; j++)//for every cluster
		{
			if (master[i].grp == j)
			{
				clusters[j].push_back(master[i]);
				break;
			}
		}
	}
}
float calculateSD(vector<float> data)
{
	float sum = 0.0;
	float mean = 0.0;
	float standardDeviation = 0.0;

	int i;

	for (i = 0; i < N; ++i)
	{
		sum += data[i];
	}

	mean = sum / N;

	for (i = 0; i < N; i++)
		standardDeviation += pow(data[i] - mean, 2);

	return sqrt(standardDeviation / N);
}
vector<Point> zScore(vector<Point> data)
{
	vector<float> SD;
	vector<float> dimension;
	vector<float> means;
	float mean = 0;
	float dev;
	int position = 0;
	for (int i = 0; i<D; i++)//for every dimension			//calculate the means and SD for every dimension
	{
		for (int j = 0; j < N; j++)//for every point
		{
			dimension.push_back(data[j].dim[i]);
			mean += data[j].dim[i];
		}
		means.push_back(mean / N);
		SD.push_back(calculateSD(dimension));
		dimension.clear();
		mean = 0;
	}
	for (int i = 0; i<D; i++)//for every dimension	//z-score normalization for every point and dimension of data
	{
		mean = means[i];
		dev = SD[i];
		for (int j = 0; j < N; j++)//for every point
		{
			if (dev == 0)
				data[j].dim[i] = 0;
			else
				data[j].dim[i] = (data[j].dim[i] - mean) / dev;
		}
	}
	return data;
}
vector<Point> read(string fileName)
{
	vector<Point> data;
	int position = 0;
	double number;
	int group;
	name = fileName;
	ifstream fin(fileName);
	if (fin.is_open())
	{
		fin >> N;
		fin >> D;
		fin >> K;
		D = D - 1;
		for (int i = 0; i < N; i++)
		{
			Point point;
			for (int j = 0; j < D; j++)
			{
				fin >> number;
				point.dim.push_back(number);
			}
			data.push_back(point);
			fin >> group;
			data[i].grp = group;
		}
		//data = zScore(data);
	}
	else
		cout << "File is not exist" << endl << endl;
	fin.close();
	master = data;
	tru = data;
	return data;
}
void calculateCentroids(vector<Point> data, int K)
{
	for (int i = 0; i < K; i++)//for every centroid
	{
		Point point;
		for (int k = 0; k < D; k++)//for every dimension
		{
			point.dim.push_back(0.0);
		}
		for (int j = 0; j < clusters[i].size(); j++)
		{
			for (int k = 0; k < D; k++)//for every dimension
			{
				point.dim[k] = point.dim[k] + clusters[i][j].dim[k];
			}
		}
		for (int k = 0; k < D; k++)
		{
			point.dim[k] = point.dim[k] / clusters[i].size();
		}
		centroids.push_back(point);
	}
}
Point maximumMethod()
{
	double max = 0;
	double min = DBL_MAX;
	Point newCentroid;

	for (int i = 1; i < N; i++)//for every point
	{
		bool taken = false;
		Point point = master[i];
		min = DBL_MAX;
		for (int k = 0; k < centroids.size(); k++)//for every centroid
		{
			Point centroid = centroids[k];
			for (int j = 0; j < D; j++)//for every dimension
			{
				double thing = centroid.dim[j] - point.dim[j];
				point.distance = point.distance + thing * thing;
			}

			if (point.distance<min) //compare shortest distance to current
			{

				min = point.distance;
				for (int j = 0; j < centroids.size(); j++)
				{
					if (i == centroids[j].grp)
					{
						taken = true;
					}

				}
				if ((taken == false) && (max<point.distance))
				{
					max = point.distance;
					newCentroid = point;
					newCentroid.grp = i;
				}
			}
			point.distance = 0;
		}
	}
	return newCentroid;
}
void classify(int K, vector<Point> data, int p)
{
	// Fill distances of all points from p
	double shortest = DBL_MAX;
	double distance = 0;
	for (int i = 0; i < K; i++) //for every centroid
	{
		Point centroid=centroids[i];
		for (int j = 0; j < D; j++)//for every dimension
		{
			double thing = centroid.dim[j] - master[p].dim[j];
			distance += thing * thing;
		}

		if (distance < shortest)//compare shortest distance to current
		{
			shortest = distance;//if shorter, update shortest
			master[p].grp = i;//update group
			master[p].distance = distance;
		}
		distance = 0;
	}
}
double calculateSSE(vector<Point> data, int K)
{
	double sum = 0.0;
	for (int i = 0; i < K; i++)
	{
		for (int j = 0; j < clusters[i].size(); j++)
		{
			sum += clusters[i][j].distance;
		}
	}
	return sum;
}
double knn(vector<Point> data, int K, int I, double T, int R, string filename)
{
	int random;
	srand(time(NULL));
	double best = DBL_MAX;
	double initial;
	int bestit;
	int run = 0;
	double bestr = DBL_MAX;
	double initialr;
	int bestitr;
	int runr = 0;
	ofstream myfile;
	myfile.open(filename);
	double SSEBestInitial;
	bestjaccards = DBL_MAX;
	bestrand = DBL_MAX;

	for (int r = 0; r < R; r++)
	{
		centroids.clear();
		centroids.push_back(master[0]);
		best = DBL_MAX;
		for (int k = 0; k < K; k++)
		{
			int number;
			number = rand() % data.size();
			centroids.push_back(data[number]);
			Sleep(1);
		}
		double SSEnew = DBL_MAX;
		double SSEold = DBL_MAX;
		int i = 1;
		for (int j = 0; j < master.size(); j++)//for every line of data
		{
			classify(K, master, j);
		}
		groupClusters(K);
		SSEnew = calculateSSE(master, K);
		cout << "Initial SSE: " << SSEnew << endl;
		initial = SSEnew;
		do
		{
			SSEold = SSEnew;
			centroids.clear();
			calculateCentroids(master, K);
			for (int j = 0; j < master.size(); j++)//for every line of data
			{
				classify(K, master, j);
			}
			groupClusters(K);
			i++;
			SSEnew = calculateSSE(master, K);
		}while(((SSEold - SSEnew) / SSEold > T) && (i < I));
		EV();
		if (SSEnew < best)
		{
			best = SSEnew;
			SSEBestInitial = initial;
		}
		//myfile << SSEnew << endl; 
		cout << "Number of iterations: " << i << endl << "Best SSE: " << best << endl << endl;
		if (bestr > best)
		{
			bestr = best;
			runr = r;
			bestitr = i;
			initialr = SSEBestInitial;
		}
		//cout << bestr << endl;
	}
	output3 << "For " << name << endl << "Best Jaccards: " << bestjaccards << endl << "Best Rands: " << bestrand<<endl<<endl;
	//output << "Best Initial SSE: " << initialr << endl;

	//output << "Number of iterations: " << bestitr << endl << "Best SSE: " << bestr << endl << endl;
	myfile.close();
	return bestr;
}
void CH(vector<Point> data, int K, int I, double T, int R, string filename)
{
	
	int k_max = sqrt(N/2);
	double St = knn(data, 1, I, T, R, filename);
	double Sw, Sb;
	double bestTrace = 0;
	double newTrace = 0;
	int bestk;
	for (int k = 2; k < k_max; k++)
	{
		Sw = knn(data, k, I, T, R, filename);
		Sb = St - Sw;
		newTrace = (N - k) /( k - 1.0)*(Sb / Sw);
		if (newTrace > bestTrace)
		{
			bestTrace = newTrace;
			bestk = k;
		}
		cout << "CH(k) for " << name << " is k= " << k << " with a value of " << newTrace << endl << endl;
	}
	
	output2 << "Best CH(k) for " << name << " is k= " << bestk << " with a value of " << bestTrace << endl << endl;
}
void SW(vector<Point> data, int K, int I, double T, int R, string filename)
{
	int k_max = sqrt(N / 2);
	int total=0;
	double bestTrace = 0;
	double newTrace = 1;
	int bestk=0;
	vector<double> averages;
	for (int k = 2; k < k_max; k++)//for every k value
	{
		double sum=0;
		knn(data, k, I, T, R, filename);
		for (int i = 0; i < k; i++)//for every cluster
		{
			for (int j = 0; j < clusters[i].size(); j++)//for every point in the cluster
			{
				for (int l = 0; l < clusters[i].size(); l++)//for every point in the cluster that isnt this point
				{
					double d1 = 0; 
					double d2=0;
					double lowest = DBL_MAX;
					if (j != l)
					{
						total++;
						for (int m = 0; m < D; m++)//for every dimension
						{
							double thing = clusters[i][j].dim[m] - clusters[i][l].dim[m];
							d1 += thing * thing;
						}
					}
					total = 0;
					for (int m = 0; m < k; m++)//for every centroid
					{
						for (int n = 0; n < D; n++)//for every dimension
						{
							double thing = clusters[i][j].dim[n] - centroids[m].dim[n];
							d2 += thing * thing;
							total++;
						}
						d2 = d2 / total;
						total = 0;
						if (d2 < lowest)
						{
							lowest=d2;
						}
					}
					double max = max(d1 , lowest);
					averages.push_back((d1 - lowest) / max);
				}
			}
		}
		for (int i = 0; i < averages.size(); i++)
		{
			sum += averages[i];
		}
		sum=sum / N;
		if (sum > bestTrace)
		{
			bestTrace = sum;
			bestk = k;
		}
		cout << "SW(k) for " << name << " is k= " << k << " with a value of " << sum << endl << endl;
		averages.clear();
	}
	output2 << "Best SW(k) for " << name << " is k= " << bestk << " with a value of " << bestTrace << endl << endl;
}
void DB(vector<Point> data, int K, int I, double T, int R, string filename)
{
	int k_max = sqrt(N / 2);
	double bestTrace = DBL_MAX;
	int bestk;
	for (int k = 2; k < k_max; k++)//for every k value
	{
		vector<double> mindistances;
		knn(data, k, I, T, R, filename);
		double minthing;
		for (int i = 0; i < k; i++)//for every cluster
		{
			double thing = sqrt(calculateSSE(clusters[i], k)/clusters[i].size());
			mindistances.push_back(DBL_MAX);
			for (int j = 0; j < k; j++)//for every cluster that isnt i cluster
			{
				double distance = 0;
				if (j != i)
				{
					double deal = thing;
					for (int d = 0; d < D; d++)//for every dimension
					{
						double blank = centroids[i].dim[d] - centroids[j].dim[d];
						distance+=blank * blank;
					}
					deal = deal / distance;
					if (deal < mindistances[i])
					{
						mindistances[i] = thing;
					}
				}
				
			}
		}
		double sum = 0;
		for (int i = 0; i < k; i++)
		{
			sum += mindistances[i];
		}
		double newTrace=sum / k;
		if (newTrace < bestTrace)
		{
			bestTrace = newTrace;
			bestk = k;
		}
		cout << "DB(k) for " << name << " is k= " << k << " with a value of " << newTrace << endl << endl;
	}

	output2 << "Best DB(k) for " << name << " is k= " << bestk << " with a value of " << bestTrace << endl << endl;
}

int main (int argc, char* argv[])
{
	output.open("total_results.txt");
	output2.open("p3results.txt");
	output3.open("p4results.txt");
	read("ecoli.txt");
	knn(master, K, 100, .001, 100, "ecoli_results.txt");
	read("ionosphere.txt");
	knn(master, K, 100, .001, 100, "ecoli_results.txt");
	read("iris_bezdek.txt");
	knn(master, K, 100, .001, 100, "ecoli_results.txt");
	read("ruspini.txt");
	knn(master, K, 100, .001, 100, "ecoli_results.txt");
	read("wine.txt");
	knn(master, K, 100, .001, 100, "ecoli_results.txt");
	read("yeast.txt");
	knn(master, K, 100, .001, 100, "ecoli_results.txt");
	read("landsat.txt");
	knn(master, K, 100, .001, 2, "ecoli_results.txt");
	read("letter_recognition.txt");
	knn(master, K, 100, .001, 2, "ecoli_results.txt");
	read("mfeat-fou.txt");
	knn(master, K, 100, .001, 2, "ecoli_results.txt");
	read("optdigits.txt");
	knn(master, K, 100, .001, 2, "ecoli_results.txt");

	//knn(read(argv[1]), atoi(argv[2]), atoi(argv[3]),atof(argv[4]),atoi(argv[5]), argv[6]); 
	//SW(read(argv[1]), atoi(argv[2]), atoi(argv[3]), atof(argv[4]), atoi(argv[5]), argv[6]);
	/*CH(read("ecoli.txt"), 6, 100, .001, 1, "ecoli_results.txt");
	SW(read("ecoli.txt"), 6, 100, .001, 1, "ecoli_results.txt");
	DB(read("ecoli.txt"), 6, 100, .001, 1, "ecoli_results.txt");
	CH(read("glass.txt"), 6, 100, .001, 1, "glass_results.txt");
	SW(read("glass.txt"), 6, 100, .001, 1, "glass_results.txt");
	DB(read("glass.txt"), 6, 100, .001, 1, "glass_results.txt");
	CH(read("ionosphere.txt"), 2, 100, .001, 1, "ionosphere_results.txt");
	SW(read("ionosphere.txt"), 2, 100, .001, 1, "ionosphere_results.txt");
	DB(read("ionosphere.txt"), 2, 100, .001, 1, "ionosphere_results.txt");
	CH(read("iris_bezdek.txt"), 3, 100, .001, 1, "iris_bezdek_results.txt");
	SW(read("iris_bezdek.txt"), 3, 100, .001, 1, "iris_bezdek_results.txt");
	DB(read("iris_bezdek.txt"), 3, 100, .001, 1, "iris_bezdek_results.txt");
	//knn(read("landsat.txt"), 6, 100, .001, 1, "landsat_results.txt");
	//knn(read("letter_recognition.txt"), 26, 100, .001, 1, "letter_recognition_results.txt");
	//knn(read("segmentation.txt"), 7, 100, .001, 1, "segmentation_results.txt");
	CH(read("vehicle.txt"), 4, 100, .001, 1, "vehicle_results.txt");
	SW(read("vehicle.txt"), 4, 100, .001, 1, "vehicle_results.txt");
	DB(read("vehicle.txt"), 4, 100, .001, 1, "vehicle_results.txt");
	CH(read("wine.txt"), 3, 100, .001, 1, "wine_results.txt");
	SW(read("wine.txt"), 3, 100, .001, 1, "wine_results.txt");
	DB(read("wine.txt"), 3, 100, .001, 1, "wine_results.txt");
	CH(read("yeast.txt"), 10, 100, .001, 1, "yeast_results.txt");
	SW(read("yeast.txt"), 10, 100, .001, 1, "yeast_results.txt");*/
	return 0;
}

