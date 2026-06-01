#ifndef NODE_H
#define NODE_H

#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <iostream>
#include <ostream>
#include <assert.h>

#include "Instance.h"
#include "globals.h"
#include "utils.h"

class Node {
public:
    // Indeksi zadnjeg meča za svaku sekvencu (-1 = prije početka)
    std::vector<int> pos;

    // Trenutno izgrađeni LCS (ako se ne rekonstruiše preko parent-a)
    std::string seq;
    // Pokazivač na roditeljski čvor
    Node* parent;  
    double score;        // vrijednost heuristike 
    double second_score;
    //feature vector is stored here after calculation in LBS
    std::vector<double> features;

    // Konstruktor
    Node(const std::vector<int>& pos_,
         const std::string& seq_,
         Node* parent_ = nullptr, double score_ = 0.0, double _second_score = 0.0)
        : pos(pos_), seq(seq_), parent(parent_), score(score_), second_score(_second_score) {}

    // Ekvivalent @property length
    int length() const {
        return static_cast<int>(seq.size());
    }
    
    static bool basic_root_node(Node *node)
    {
        for(int x: node->pos)
            if( x != -1)
                return false;
                
        return true;
    }

    // ============================================================
    // Generate child nodes (ekvivalent generate_successors)
    // ============================================================
    static std::vector<Node*> generateSuccessors(
        Node* node,
        Instance* inst) {
        const std::vector<std::string>& sequences = inst->sequences;
        const std::vector<std::vector<int>>& gaps = inst->gaps;
        const int m = sequences.size();

        // per_seq_maps[i] : map<char, index>
        std::vector<std::unordered_map<char, int>> perSeqMaps(m);
        
        bool root_node = false;
        if(basic_root_node(node))
        {
                root_node = true;
                int S = sequences.size();
                std::unordered_map<char, std::vector<int>> first_positions; 
                
                
                for (int c = 0; c < (int)inst->Sigma.size(); ++c) { // for each letter
                    char a = inst->Sigma[c];
                    
                    std::vector<int> pos(S, -1);
                    for (int i = 0; i < S; ++i) { // the positions of its first occurances in search s_i are detected and stored 
                         for(int j = 0; j < (int)sequences[i].size(); ++j)
                             if(sequences[i][j] == a){
                                 pos[i] = j;
                                 break;
                             }
                   }
               
                   first_positions[a] = pos;
                }
                
                for (int i = 0; i < m; ++i)
        	{
                     std::unordered_map<char, int> map;
                     for(char ch: inst->Sigma) 
                          map[ch] =  first_positions[ch][i]; 
                     
                     perSeqMaps[i] = map;         
                }      
        }
        else{
          
        	for (int i = 0; i < m; ++i)
        	{
            		if(node->pos[i] + 1 >= (int)sequences[i].size())
            	    		return {};
            
            		std::unordered_map<char, int> map;
            
            		for(char ch: inst->Sigma) {        
                		map[ch] = inst->Succ[i][inst->charToInt[ch]][node->pos[i] + 1]; // start with pos[i] + 1 for seeking matched chars
            	        }  
            	        perSeqMaps[i] = map;
               }            
        }
        
        //for(int i=0; i<perSeqMaps.size(); ++i)
        //    for(auto & x: perSeqMaps[i])
        //        std::cout << x.first << " " << x.second << std::endl;

        
        // --------------------------------------------------------
        // Presjek karaktera koje sve sekvence mogu mečovati
        // --------------------------------------------------------
        std::set<char> commonChars;
        for(char ch: inst->Sigma) {
                // pitaj da li se ch nalazi u svim mapama (index i odozgo) i da je različit od -1
                bool found_in_all = true;
                for (int i = 0; i < m; ++i) {  
                    if (perSeqMaps[i].find(ch) == perSeqMaps[i].end() ||
                        perSeqMaps[i][ch] == -1) {
                        found_in_all = false;  
                        break;
                    }
                }
                if (found_in_all) {
                    commonChars.insert(ch);
                }
        }

        if (commonChars.empty()) // this is the complete node 
            return {};

        // --------------------------------------------------------
        // Removal of dominated characters from @commonChars
        // --------------------------------------------------------

        std::set<char> dominated;

        for (char c : commonChars) {
            for (char o : commonChars) { //run through the pairs of different characters (c, o)
                if (c == o) continue; // symmetry breaking

                bool dominates = true;
                for (int i = 0; i < m; ++i) {
                    int dc = perSeqMaps[i].at(c);
                    int do_ = perSeqMaps[i].at(o);
                    if (!(dc <= do_ && 
                          (do_ - dc  <= gaps[i][do_] or root_node))) {  // breaking symmetric condition
                        dominates = false;
                        break;
                    }
                }
                if (dominates)
                    dominated.insert(o);
            }
        }
  
        for (char d : dominated) // remove dominated chars from commonChars
            commonChars.erase(d);
    
        std::vector<Node*> successors;
        for (char ch : commonChars) {
            std::vector<int> idxVector(m);

            for (int i = 0; i < m; ++i){
                
                idxVector[i] = perSeqMaps[i].at(ch); // new vector of positions 
                //std::cout << idxVector[i] << " " << node->pos[i] << " " << gaps[i][ idxVector[i] ] + 1 << std::endl;
                assert(idxVector[i] - node->pos[i] <= gaps[i][ idxVector[i] ] + 1 or node->pos[i] == -1 ); // gap constraint check
            }   
  
            Node* child = new Node(
                idxVector,
                node->seq + ch,// to optimize
                node
            );
            
            //child->print();

            successors.push_back(child);
        }

        return successors;
    }

// Generate successors in BACKWARD manner
// (equivalent to generate_backward_successors in Python)
// ============================================================


static std::vector<Node*> generateBackwardSuccessors(
    Node* node, Instance* inst) {
    
    const std::vector<std::string>& sequences = inst->sequences;
    const std::vector<std::vector<int>>& gaps = inst->gaps;

    const int m = sequences.size();
    std::vector<Node*> successors;

    for (int ch = 0; ch < (int)inst->Sigma.size(); ++ch) {
        
        char a = inst->Sigma[ ch ];
        std::vector<int> prev_positions;
        bool feasible = true;

        for (int i = 0; i < m; ++i) {
            int pi = node->pos[i];

            if (pi < 0) { // complete path when ging backwards 
                feasible = false;
                break;
            }
            
            const std::vector<int>& prev_vec =  inst->Prev[i][ ch ];
            if (pi >= (int)(prev_vec.size())) {
                feasible = false;
                break;
            }
            
            // If some index is -1, not feasible
            if (prev_vec[pi] == -1){  // no feasible extension
                feasible = false;
                break;  
            }
            // if everything all right, check the gap constraint fulfillment (4, 5) --> (3, 2)
            int pa = prev_vec[pi];
            if (pi - pa > gaps[i][pi] + 1) {
                
                feasible = false;
                break;
            }

            prev_positions.push_back(pa);
        }

        if (!feasible)
            continue;

        // build child
        Node* child = new Node(
            prev_positions,
            node->seq + a,   // backward → append
            node
        );  

        successors.push_back(child);
    }

    return successors;
}
    // ====================GENERATE SUCCESORS in an LCS manner ========================================
    
    static std::vector<Node*> generateSuccessorsLCS(Node* node, Instance* inst, HeuristicType heuristic) {
    
         int index = 0; std::vector<Node*> child_nodes;
         // check if the node not (trivially) complete
         for(int px: node->pos)
         {
             if (px >= (int)inst->sequences[index].size()) 
                 return {};
             index++;
         }
         // determine the positions of next occurrences of each letter starting from the positions of node @node 
         std::set<char> NotCommonChars;
         std::unordered_map<char, std::vector<int>> next_char_positions;
         for(char ch: inst->Sigma)
         {
              std::vector<int> char_positions = std::vector<int>(inst->sequences.size(), -1); // all -1s initialized
              for(int i=0; i < (int)inst->sequences.size(); ++i)
              {
                  int next_pos = -1;
                  for(int j = node->pos[i] + 1; j < (int)inst->sequences[i].size(); ++j)
                  {
                      if(inst->sequences[i][j] == ch)
                      {
                          next_pos = j;    
                          break;    
                      }
                  }
                  if(next_pos == -1)
                      NotCommonChars.insert(ch);
                  
                  char_positions[i] = next_pos;
              }    
              next_char_positions[ch] = char_positions; 
         }
         // next_char_positions  
         std::set<char> CommonChars;
         for(char ch: inst->Sigma)
             CommonChars.insert(ch);
         
         //remove letters from 
         for(char c: NotCommonChars)
         {
             auto it = CommonChars.find(c);
             if(it != CommonChars.end())
                 CommonChars.erase(it);
         }
         // Now create nodes 
         for(char ch: CommonChars)
         {
             Node* child = new Node(next_char_positions[ch], "", nullptr);
             child->evaluate(inst, heuristic);
             child_nodes.push_back(child);
         }
         
         return child_nodes;
    }

    double evaluate(Instance* inst, HeuristicType heuristic, int k = 0, bool forward_or_backward = true) { // required to pass for forward and backward BS
    
    
    	if (not forward_or_backward) // backward BS (guided by H5) 
    	{
    	    this->score = h5_backward(this->pos, inst->C_suffix, inst->Sigma); 

    	}else{
              
              std::string heur = heuristicToString(heuristic);
              if( heuristicToString(HeuristicType::H1) == heur )
              {
                           this->score = this->length(); //lv  

              }
               if( heuristicToString(HeuristicType::H2) == heur )
               {
                           this->score = remaining_lb(this->pos, inst->sequences); 
                           this->score += this->length(); //add the length of partial solution associated to the node 
               }

              if( heuristicToString(HeuristicType::H5) == heur )
              {
                           this->score = h5_upper_bound(this->pos, inst->C_suffix,
                                   inst->Sigma,
                                   inst->sequences); 
                           this->score += this->length(); //add the length of partial solution associated to the node              
              }
              if( heuristicToString(HeuristicType::H8) == heur)
              {           
              
 
                          this->score =  probability_based_heuristic(this->pos, inst->P, inst->sequences, k);  
                          //std::cout << " " << this->score << std::endl;
              }
      }
    return this->score;
}
  

void print(std::ostream& os = std::cout) {
    os << "Node {\n";
    os << "  seq   = \"" << seq << "\"\n";
    os << "  len   = " << seq.size() << "\n";
    os << "  pos   = [ ";
    for (int p : pos)
        os << p << " ";
    os << "]\n";
    os << "  score = " << score << "\n";
    os << "}";
}  


static std::unordered_map<char, int> firstValidIndices(
        const std::string& seq,
        int startIdx,
        const std::vector<int>& gapArr,
        int prevIdx
    ) {
        std::unordered_map<char, int> mapping;

        for (int j = startIdx; j < static_cast<int>(seq.size()); ++j) {
            if (prevIdx != -1 && (j - prevIdx) > gapArr[j] + 1)
                continue;

            char c = seq[j];
            if (mapping.find(c) == mapping.end()) {
                mapping[c] = j; // prvi (najlijevlji) hit
            }
        }
        return mapping;
    }
};

#endif // NODE_H

