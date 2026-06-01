#ifndef BEAMSEARCH_H
#define BEAMSEARCH_H

#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include <chrono>
#include <unordered_map>

#include "Node.h"
#include "utils.h"
#include "Instance.h"
#include "nnet.h"
#include <Eigen/Dense>
#include "unistd.h"

class MLP;

class BeamSearch {
public:

    struct Result {
        std::string best_seq;
        std::vector<std::vector<int>> steps;
        double runtime;
        std::vector<std::vector<int>> list_pos_complete;
    };
    
    struct VectorHash {
      std::size_t operator()(const std::vector<int>& v) const noexcept {
        std::size_t h = 0;
        for (int x : v) {
            h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
      }
   };


static void write_result(
    std::ostream& out,
    const Result& res,
    const Instance& inst,
    const std::string& title
) {
    out << "=== " << title << " ===\n";
    out << "Best sequence: " << res.best_seq << "\n";
    out << "Length       : " << res.best_seq.size() << "\n";
    out << "Runtime (s)  : " << res.runtime << "\n";

    // feasibility check
    bool feasible = check_feasibility(res.steps, inst.gaps, true);
    out << "Feasible     : " << (feasible ? "YES" : "NO") << "\n";

    out << "Steps:\n";
    for (const auto& step : res.steps) {
        out << "( ";
        for (int p : step)
            out << p << " ";
        out << ") ";
    }
    out << "\n";
}

   
   static void compute_heuristic_values(std::vector<Node*>& V_ext,  MLP* neural_network, Instance* inst){    
      
       for(Node* node : V_ext){
           // forward pass: assign a heuristic value from NN to each node 
          if(not node->features.empty())
     {
	           Eigen::Map<const Eigen::VectorXd> eigen_features(node->features.data(), node->features.size());
               node->score = neural_network->forward(eigen_features)(0);
               //std::cout << "Node score from NN: " << node->score << std::endl;
               double primary_score = node->score;
               node->evaluate(inst, HeuristicType::H5);
               node->second_score =  node->score;
               node->score = primary_score;
           }else{
               node->second_score = node->evaluate(inst, HeuristicType::H5);//node->length();

	           node->score=-10000;//node->length();
	   }	   
           //}
      }
  }


  static void compute_features(std::vector<Node*>& V_ext, Instance* inst)
  {
    //calculate features for the nodes in V_ext
    for(Node* node : V_ext)
    {
        //first option: consider p^{L,v} and l^v. first do p^{L,v}_i = p^{L,v}_i / |s_i| and l^v_j = l^v_j / |r_j| for al input and restricted strings. 
        // then consider the max, min, std, avg of both vectors as features (8 features). Add also length of partial sol represented by node u^v (9 in total) (config1)
        //Add the alphabet size (config2)
        //Add length and number of input and restricted strings (config3)
        std::vector<double> pL_v; int index=0; bool complete_node=false;
        pL_v.reserve(node->pos.size());
        for (int x : node->pos)
	{	
	    if(x >= (int)inst->sequences[index].size()-1)
		  complete_node=true;		  
	    
            pL_v.push_back(static_cast<double>(x));
            
	   index++; 
	}
	// determine the features;

	int len_partial = node->length();
        
        //normalize left position vectors so that they do not depend on the length of the input strings
        for(int i = 0; i < (int)pL_v.size(); ++i) pL_v[i] /=  inst->sequences[i].size();
        
	if(complete_node)
          continue;

        vector<double> features;
       	// pL_v stats (NO vector)
       	
	double min_p = 1e18, max_p = -1e18;double mean_p = 0.0, M2_p = 0.0;int count_p = 0;
	for (int i = 0; i < (int)node->pos.size(); ++i) {   
	       	double val = (double)node->pos[i] / inst->sequences[i].size();   
	       	min_p = std::min(min_p, val);   
	       	max_p = std::max(max_p, val);   
	       	count_p++;   
	       	double delta = val - mean_p; 
	     	mean_p += delta / count_p;  
	      	M2_p += delta * (val - mean_p);
	  
        }
	double std_p = (count_p > 1) ? std::sqrt(M2_p / count_p) : 0.0;
       	features.push_back(max_p);
	features.push_back(min_p);
	features.push_back(mean_p);
	features.push_back(std_p);
	features.push_back(len_partial);
	
	// ---- gaps stats (NO vector) 
	double min_g = 1e18, max_g = -1e18;double total_sum = 0.0;double total_sq_sum = 0.0;int total_count = 0;
	int m = inst->sequences.size();
	for (int i = 0; i < m; ++i) {   
	     int start = node->pos[i] + 1;    
	     int n = inst->sequences[i].size();   
	     if (start >= n) continue;   
	     int count = n - start;    
	     total_sum += inst->suffix_sum[i][start];   
	     total_sq_sum += inst->suffix_sq_sum[i][start];      
	     min_g = std::min(min_g, inst->suffix_min[i][start]);     
	     max_g = std::max(max_g, inst->suffix_max[i][start]);    
	     total_count += count;
	}
	// finalize stats
    double mean_g = (total_count > 0) ? total_sum / total_count : 0.0;
	double variance = 0.0;
	if (total_count > 0) {    
		variance = (total_sq_sum / total_count) - (mean_g * mean_g);}
	double std_g = (variance > 0.0) ? std::sqrt(variance) : 0.0;
	// push features
	features.push_back(max_g);
	features.push_back(min_g);
	features.push_back(mean_g);
	features.push_back(std_g);
	
	/*vector<double> features; 
        //features of vectors pL_v and lv
        double mean_pL_v = compute_average(pL_v);
          
        features.push_back(compute_max(pL_v));
        features.push_back(compute_min(pL_v));
        features.push_back(mean_pL_v);
        features.push_back(compute_std(pL_v, mean_pL_v));
        features.push_back(len_partial);
       
        //gaps features
        std::vector<double> gaps_all_features; //gaps_all_features.push_back(10000);
        for(int i = 0; i < (int)inst->sequences.size(); ++i){

                for(int j = node->pos[i]+1; j < (int)inst->sequences[i].size(); ++j)
                    gaps_all_features.push_back(static_cast<double>(inst->gaps[i][j]));
                
        }
		
        double mean_gaps = compute_average(gaps_all_features);
        features.push_back(compute_max(gaps_all_features));
        features.push_back(compute_min(gaps_all_features));
        features.push_back(mean_gaps);
        features.push_back(compute_std(gaps_all_features, mean_gaps));          
        //additional features based on configuration (for now 9 of them in total)         
        */

        if(feature_config == 2) // add alphabet size, 10 features 
            features.push_back( (int)inst->Sigma.size() );
            
        else if(feature_config == 3){ // 11 features
            
            features.push_back( (int)inst->Sigma.size() );
            features.push_back( (int) inst->sequences.size() ); // number of instance 
        }
        else if(feature_config == 4){ // 12 features
        
            features.push_back( (int)inst->Sigma.size() );  
            features.push_back( (int) inst->sequences.size() );
            features.push_back((inst->sequences[0]).size()); //note that this config only makes sense if all input strings have the same length
        }
        standardize(features);

        node->features = features;
      }
   }

    static Result run_forward_backward_BS(
        Instance* inst,
        bool forward_or_backward,
        int beam_width = 10,
        HeuristicType heuristic = HeuristicType::H5,  // forward or backward heuristic, depending on the parameter @formward_or_backward
        int time_limit_sec = 1800, // max time allowed is 30 mins 
        const std::vector<Node*>& start_node_vector = {},
        MLP* neural_network = nullptr  
    ) {
        
        const auto& sequences = inst->sequences;
        //const auto& gaps = inst->gaps;
        const int m = sequences.size();
        std::vector<Node*> all_nodes; // trace all nodes, at the end delete them all
        
        std::vector<std::vector<int>> list_pos_complete;
        std::vector<Node*> beam;
        std::vector<int> init_pos(m, -1);
        int max_iters = 100000; // should be parametrized 
        
        Node* start_node = nullptr;

        if(start_node_vector.size() == 0 ){
            start_node = new Node(init_pos, "", nullptr); // start from the beginning
            beam.push_back(start_node);
        }
        else{
             for(Node* n: start_node_vector){
                 beam.push_back(n); 
                 //if(forward_or_backward) // if just forward BS, add the start nodes to all_nodes to be deleted at the end
                 //    all_nodes.push_back(n);
             }  
        }
 
        std::string best_seq;
        Node* return_node = nullptr;

        auto time_start = std::chrono::steady_clock::now();

        for (int iter = 0; iter < max_iters && !beam.empty(); ++iter) 
        {
            
            if (std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - time_start
                ).count() > time_limit_sec)
                break;
            
            std::vector<Node*> candidates;
            for (Node* nx : beam) 
            {
                
                auto succs = forward_or_backward
                    ? Node::generateSuccessors(nx, inst)    
                    : Node::generateBackwardSuccessors(nx, inst);
                
                if (succs.size()==0){ 
                    list_pos_complete.push_back(nx->pos);
                    continue; 
                }
                for (Node* s : succs) {
                    
                    candidates.push_back(s); all_nodes.push_back(s);
                    
                    if (s->seq.size() > best_seq.size()) {
                        best_seq = s->seq;
                        return_node = s;
                    }
                }
            }                  //std::cout << "candidates.empty(): " << candidates.empty() << std::endl;

            if (candidates.empty()) break;
            
            // if heuristic H8 is used, determine k value here
            int k_val = 0;
            if(heuristic == HeuristicType::H8 and forward_or_backward and neural_network == nullptr) // prob-guidance only applied to the forward BS manner 
            {
                 int min_len = 1000000;
                 for (Node* n : candidates){
                     for(int i=0; i < (int)inst->sequences.size(); ++i)
                         if((int)inst->sequences[i].size() - n->pos[i]+1 < min_len)
                             min_len = inst->sequences[i].size() - n->pos[i] + 1;                
                 }
                 k_val =  (double) min_len / (int)inst->Sigma.size() > 1 ?  (int)((double) min_len / (int)inst->Sigma.size() )  : 1;
            }
            
            //  Node evaluation here is assigned 
            if(neural_network == nullptr)
            
                for (Node* n : candidates) {
                     n->evaluate(inst, heuristic, k_val, forward_or_backward); 
                }
            
	    else{ //use the outcome from NN as heuristic guidance  
                 compute_features(candidates, inst);
                 compute_heuristic_values(candidates, neural_network, inst); 
            }
                
	     
	    // sort candidates by their heuristic values and keep the top ones according to the beam width
            std::sort(candidates.begin(), candidates.end(),
                    [](Node* a, Node* b) { return a->score > b->score; });


                //Store the rank of each node and then sort the nodes (in the candidate list) accroring to h5 and also report the rank of nodes...
                // then determine the average rank of each node, sort the list according to the average rank and keep the top ones according to the beam width
                
		if (lambda < 1.0) // if not in the trained process: if lambda=0.0, then UB only counted, if lambda=1.0, only data-driven heuristic counted 
                {
            
                    std::unordered_map<Node*, int> rank_first = std::unordered_map<Node*, int>();
                    for(int i=0; i<(int)candidates.size(); ++i)
                        rank_first[candidates[i]] = i+1; //rank starts from 1     
            
                    // sort by UB (second_score) and determine the rank of each node according to the UB values
                    std::sort(candidates.begin(), candidates.end(),
                        [](Node* a, Node* b) { return a->second_score > b->second_score; });
             
                    std::unordered_map<Node*, int> rank_second = std::unordered_map<Node*, int>();
                    for(int i=0; i<(int)candidates.size(); ++i)
                        rank_second[candidates[i]] = i+1; //rank starts from 1  

                    std::sort(candidates.begin(), candidates.end(), 
                        [&rank_first, &rank_second](Node* a, Node* b) { 
                            double avg_rank_a = (lambda * rank_first.at(a) + (1-lambda) * rank_second.at(a));
                            double avg_rank_b = (lambda * rank_first.at(b) + (1-lambda) *  rank_second.at(b));
                                return avg_rank_a < avg_rank_b; // sort by average rank (lower is better)       
                    });
                }   
            
            if ((int)candidates.size() > beam_width)
                candidates.resize(beam_width);

            beam = candidates;
        }

    //constructing solution steps
	std::vector<std::vector<int>> steps;

	if (return_node != nullptr) {
            //extract the state sequence from the return_node
    	    while (return_node != nullptr) 
            {   
                 if (return_node->pos[0] != -1)
        	          steps.push_back(return_node->pos);
                 else 
                        break; // reached the root node
                 return_node = return_node->parent;
            }
            if(not forward_or_backward) //if done backwardly
            {    
                 //if(start_node_vector.size() != 0) // when we do not start with the basic root node (-1, ... , -1), add letter at the beginning position also to the solution 
                 if(start_node_vector[0]->pos[0] >= 0)
                    best_seq =  inst->sequences[0][ steps[steps.size()-1][0] ] + best_seq; //add the starting (matched) letter
            }
            
            if (forward_or_backward) // if not backward, no need to reverse back the @str value, only the collected steps
    	        std::reverse(steps.begin(), steps.end());
    	    else
    	        std::reverse(best_seq.begin(), best_seq.end()); // other
        }

        double runtime = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - time_start
        ).count();
        
        // clean up the memory(release the occupied memory)
        for (Node* n : all_nodes)
        if (n != nullptr)
            delete n;

        return { best_seq, steps, runtime, list_pos_complete };
    }
    
    // IMSBS algorithm (adapt for NN )  
    static Result imsbs_with_learning(
        
        Instance* inst,
        int beam_width_forward = 10,
        int beam_width_backward = 10,
        HeuristicType heuristic_forward = HeuristicType::H5,
        int number_root_nodes = 10,
        int imsbs_iterations = 10000,  
        int time_limit_sec = 1800,  MLP* neural_network = nullptr ) {
        
            //print values of all parameters passed to the function
        /*std::cout << "IMSBS parameters: beam_width_forward=" << beam_width_forward << ", beam_width_backward=" << beam_width_backward
                  << ", heuristic_forward=" << static_cast<int>(heuristic_forward) << ", number_root_nodes=" << number_root_nodes
                  << ", imsbs_iterations=" << imsbs_iterations << ", time_limit_sec=" << time_limit_sec << std::endl; -
*/
        std::vector<Node*> all_nodes; // trace all nodes, at the end delete them all
        
        int best_sol_found=0; std::string best_seq="";  //best solution attributes 
        std::vector<Node*> R; // can be also priority queue: TODO 
        Node *root_node = new Node(std::vector<int>(inst->sequences.size(), -1), "", nullptr); // initial root node
        R.push_back(root_node);  all_nodes.push_back(root_node);

        // EFFICIENT VISITED CHECK
        std::unordered_map<std::vector<int>, Node*, VectorHash> visited;
        std::vector<std::vector<int>> best_steps; double runtime=0.0;
        
        auto time_start = std::chrono::steady_clock::now();
        for(int iter = 0; iter < imsbs_iterations && (R.size() != 0); ++iter)
        {    
            //std::cout << "IMSBS iteration " << iter + 1 << std::endl;
             int r_size = std::min(static_cast<size_t>(number_root_nodes), R.size());
             
             std::vector<Node*> L;
             while ((int)L.size() < r_size) 
             {
                  L.push_back(R[0]);
                  R.erase(R.begin());
             }             
             //L initialized; refine the nodes by running backward BS per each node
             std::unordered_map<std::vector<int>, std::vector<std::vector<int>>, VectorHash> root_node_steps; 
             for(Node *n : L)
             {
                 
   	              Result res_n = run_forward_backward_BS(
        		               inst,
        		               false, // backward BS
        		               beam_width_backward,
        		               HeuristicType::H5,
        		               time_limit_sec, 
        		               {n} // run the backward BS on @n
                 );
    		      // REFINING (candidate-root) NODES 
    		      n->seq = res_n.best_seq;  
    		      const std::vector<int> root_pos = n->pos;
    		      root_node_steps.emplace(root_pos, res_n.steps);
    		      n->parent = nullptr;  
             } 
             //execute the FORWARD BS on the set of refined nodes from L (nodes from L will be discarded after this step)
             Result res_n = run_forward_backward_BS(
        			inst,
        			true, // forward BS
        			beam_width_forward,
        			heuristic_forward,
        			time_limit_sec, 
        			L, neural_network // run the forward BS on L, if NN is provided, @heuristic_forward will be ignored inside the function
             );  
             
             //Check IF new incumbent solution 
             if(best_sol_found < (int)res_n.best_seq.size())
             {
                 best_sol_found = res_n.best_seq.size();
                 best_seq =  res_n.best_seq;
                 //reconstruct steps from the backward BS 
                 best_steps = res_n.steps;
                 // push_front steps to @best_steps obtained from the backward BS pass (root_node_steps)
                 std::vector<std::vector<int>> best_step_front = best_steps.size() > 0 ? root_node_steps[best_steps[0]]  : std::vector<std::vector<int>>(); 
                 for(int i=0; i<(int)best_step_front.size()-1 && best_step_front.size() > 0; ++i) // exclude the last one as it appears in the @best_steps already 
                     best_steps.insert(best_steps.begin() + i, best_step_front[i]);   
             }
             // Update R: should pass all complete solutions during the forward BS
             for(auto& pos: res_n.list_pos_complete)
             {
                 Node* p_new = new Node(pos, "", nullptr); // create a new node as a candidate root node
                 // expand p_new 
                 auto succs = Node::generateSuccessorsLCS(p_new, inst, HeuristicType::H5); // generate in the LCS manner 
                 // add succs into R if not already been part of R (in the previous iterations)
                 for(Node* child: succs)
                 {
                     if(visited.find(child->pos) == visited.end()) // @child not in @visited
                     {
                         R.push_back(child);
                         visited.emplace(child->pos, child);
                         all_nodes.push_back(child);
                     }else
                        delete child;
                 }
                 all_nodes.push_back(p_new); // add p_new to the all_nodes to be deleted at the end
             }

             runtime = std::chrono::duration<double>(
             std::chrono::steady_clock::now() - time_start
             ).count();
             
             if(runtime >= time_limit_sec) //time has exceeded 
                 break;
             //sort out R vector by UB (makes sense as partial solutions here are different)
             if(neural_network == nullptr) // if the basic IMSBS
                 std::sort(R.begin(), R.end(), [](const Node* a, const Node* b) { return a->score > b->score; });
             else // if the learning BS included (sort by UB)
                 std::sort(R.begin(), R.end(), [](const Node* a, const Node* b) { return a->second_score > b->second_score; });
        }
        //cleanup nodes  
        
        for(Node* node: all_nodes)
            if (node != nullptr)
                delete node;
        //delete the structure @visited, all pointers  in @visited are already deleted in the all_nodes deletion step
        visited.clear();  
        
        return {best_seq, best_steps, runtime , {}};
    }
};

#endif // BEAMSEARCH_H

