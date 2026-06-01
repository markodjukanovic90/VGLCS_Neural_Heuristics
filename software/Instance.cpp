#include "Instance.h"
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <sstream>
#include <set>
#include <limits>
#include <ostream>
#include <iostream>


void Instance::buildCharToInt() {

    charToInt.clear();
    for (int i = 0; i < (int)Sigma.size(); ++i) {
        charToInt[Sigma[i]] = i;
    }
}

void Instance::buildSuffixCounts() {

    C_suffix.clear();
    C_suffix.reserve(sequences.size());

    for (const std::string& s : sequences) {
        int n = s.size(); // |s_i|

        std::unordered_map<char, std::vector<int>> Ci;

        // inicijalizacija: Ci[a] = vector<int>(n+1, 0)
        for (char a : Sigma)
            Ci[a] = std::vector<int>(n + 1, 0);

        // popunjavanje od kraja ka početku
        for (int j = n - 1; j >= 0; --j) {
            char ch = s[j];

            for (char a : Sigma)
                Ci[a][j] = Ci[a][j + 1];

            Ci[ch][j] += 1; // when C_suffix[i][a][p^v_i] = number of occurences of letter a in s_i[p^v_i :]
        }

        C_suffix.push_back(std::move(Ci));
    }
}

void Instance::buildPrevTable() {
    int S = sequences.size();
    int K = Sigma.size();

    Prev.clear();
    Prev.resize(S);

    for (int i = 0; i < S; ++i) {
        const std::string& s = sequences[i];
        int n = s.size();

        // Prev[i][c][j]
        Prev[i].assign(K, std::vector<int>(n + 1, -1));

        // last[c] = posljednja pozicija znaka c
        std::vector<int> last(K, -1);

        for (int j = 0; j <= n; ++j) {
            // kopiraj trenutne "last" vrijednosti
            for (int c = 0; c < K; ++c)
                Prev[i][c][j] = last[c];

            if (j < n) {  
                int cidx = charToInt[s[j]];
                last[cidx] = j;
            }
        }
    }
}

void Instance::buildSuccTable() {
    int S = sequences.size();
    int K = Sigma.size();

    Succ.clear();
    Succ.resize(S);

    for (int i = 0; i < S; ++i) {
        //const std::string& s = sequences[i];

        int n = sequences[i].size();
        
        // Succ[i][c][j]
        Succ[i].assign(K, std::vector<int>(n + 1, -1));

        // For each character in Sigma
        for (int c = 0; c < K; ++c) {
            char a = Sigma[c];

            // Collect all positions where s[pos] == a
            std::vector<int> positions;
            for (int pos = 0; pos < n; ++pos) { // n is the size of s_i
                if (sequences[i][pos] == a)
                    positions.push_back(pos);
            }

            // For each j, find smallest feasible x
            for (int j = 0; j <= n; ++j) {
                int best = -1;

                for (int x : positions) {
                    if (x < j) // we want indices   j or after as we move forward by at least one position 
                        continue;

                    // gap feasibility check
                    if (x - j <= gaps[i][x]) { // +1
                        best = x;
                        break;
                    }
                }

                Succ[i][c][j] = best;
            }
        }
    }      
}


void Instance::buildGapSuffixStats() 
{   
       	int S = sequences.size();
    	suffix_sum.resize(S);
       	suffix_sq_sum.resize(S);    
	suffix_min.resize(S);    suffix_max.resize(S);  
      	for (int i = 0; i < S; ++i) {   
	     int n = sequences[i].size();  
       	     suffix_sum[i].assign(n + 1, 0.0);   
	     suffix_sq_sum[i].assign(n + 1, 0.0);     
	     suffix_min[i].assign(n + 1, std::numeric_limits<double>::infinity());      
	     suffix_max[i].assign(n + 1, -std::numeric_limits<double>::infinity()); 
      	     for (int j = n - 1; j >= 0; --j) {   
	    	 double val = (double)gaps[i][j];    
	 	 suffix_sum[i][j] = suffix_sum[i][j + 1] + val;   
		 suffix_sq_sum[i][j] = suffix_sq_sum[i][j + 1] + val * val;   
		 suffix_min[i][j] = std::min(val, suffix_min[i][j + 1]);   
		 suffix_max[i][j] = std::max(val, suffix_max[i][j + 1]);        
	     }   
       	}
}

void Instance::buildPTable(int max_n) {
    int sigma = Sigma.size();
    
    //std::cout << "building P " << std::endl;
    if (sigma <= 0)
        throw std::runtime_error("Alphabet must be built before P-table");

    // allocate P[max_n+1][max_n+1]
    P.assign(max_n + 1, std::vector<double>(max_n + 1, 0.0));

    // P[i][0] = 1  
    for (int i = 0; i <= max_n; ++i)
        P[i][0] = 1.0;

    // P[0][j>0] remains 0

    for (int n = 1; n <= max_n; ++n) {
        for (int k = 1; k <= n; ++k) {
            P[n][k] =
                (P[n - 1][k - 1] + (sigma - 1) * P[n - 1][k]) / sigma;
        }
    }
}


Instance Instance::loadFromFile(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open instance file: " + filename);
    }

    Instance inst;

    // ---- read number of sequences ----
    int m;
    in >> m;
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    inst.sequences.resize(m);
    inst.gaps.resize(m);

    std::set<char> sigma_set;

    for (int i = 0; i < m; ++i) {
        // ---- read sequence ----
        std::getline(in, inst.sequences[i]);
        if (inst.sequences[i].empty()) {
            throw std::runtime_error("Empty sequence at index " + std::to_string(i));
        }

        for (char c : inst.sequences[i])
            sigma_set.insert(c);

        const int n = inst.sequences[i].size();
        inst.gaps[i].resize(n);

        // ---- read gaps ----
        for (int j = 0; j < n; ++j) {
            if (!(in >> inst.gaps[i][j])) {
                throw std::runtime_error(
                    "Not enough gap values for sequence " + std::to_string(i)
                );
            }
        }

        in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    
    inst.buildAlphabet(); 
    inst.buildCharToInt();
    inst.buildSuffixCounts();
    inst.buildPrevTable();
    inst.buildSuccTable();
    inst.buildGapSuffixStats();
    //P-table if necessary for some algorithms 
    int max_n = 0;
    for (const auto& s : inst.sequences)
        if ((int)s.size() > max_n)
            max_n = s.size();
    //std::cout <<"max_n: " << max_n << std::endl;
    inst.buildPTable(max_n);
    
    return inst;
}

void Instance::buildAlphabet() {
    std::unordered_set<char> seen;
    for (const auto& s : sequences)
        for (char c : s)
            seen.insert(c);

    Sigma.assign(seen.begin(), seen.end());
}

void Instance::print(std::ostream& os) {

    const int m = sequences.size();
    os << m << "\n";

    for (int i = 0; i < m; ++i) {
        // ---- sequence ----
        os << sequences[i] << "\n";

        // ---- gaps ----
        for (size_t j = 0; j < gaps[i].size(); ++j) {
            os << gaps[i][j];
            if (j + 1 < gaps[i].size())
                os << " ";
        }
        os << "\n";
    }
}


