#include "translator.h"

SentenceTranslator::SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen)
{
	src_vocab = i_models.src_vocab;
	tgt_vocab = i_models.tgt_vocab;
	ruletable = i_models.ruletable;
	lm_model = i_models.lm_model;
	para = i_para;
	feature_weight = i_weight;

	src_tree = new SyntaxTree(input_sen);
	src_sen_len = src_tree->sen_len;
}

SentenceTranslator::~SentenceTranslator()
{
	//TODO
}

string SentenceTranslator::words_to_str(vector<int> wids, bool drop_unk)
{
		string output = "";
		for (const auto &wid : wids)
		{
			string word = tgt_vocab->get_word(wid);
			if (word != "NULL" || drop_unk == false)
			{
				output += word + " ";
			}
		}
		TrimLine(output);
		return output;
}

vector<TuneInfo> SentenceTranslator::get_tune_info(size_t sen_id)
{
	vector<TuneInfo> nbest_tune_info;
	//TODO
	return nbest_tune_info;
}

vector<string> SentenceTranslator::get_applied_rules(size_t sen_id)
{
	vector<string> applied_rules;
	//TODO
	//dump_rules(applied_rules,best_cand);
	return applied_rules;
}

/**************************************************************************************
 1. 函数功能: 获取当前候选所使用的规则
 2. 入口参数: 当前候选的指针
 3. 出口参数: 用于记录规则的applied_rules
 4. 算法简介: 通过递归的方式回溯, 如果当前候选没有子候选, 则找到了一条规则, 否则获取两个
 			  子候选所使用的规则
************************************************************************************* */
void SentenceTranslator::dump_rules(vector<string> &applied_rules, Cand *cand)
{
	//TODO
}

string SentenceTranslator::translate_sentence()
{
	if (src_sen_len == 0)
		return "";
	for (size_t span=0;span<src_sen_len;span++)
	{
//#pragma omp parallel for num_threads(para.SPAN_THREAD_NUM)
		for(size_t beg=0;beg<src_sen_len-span;beg++)
		{
			int end = beg + span;
			auto it = src_tree->nodes_at_span.find(beg<<16 + end);
			if ( it == src_tree->nodes_at_span.end() )
				continue;
			else
			{
				for (const auto node : it->second)
				{
					generate_kbest_for_node(node);
				}
			}
		}
	}
	//TODO
	//return words_to_str(src_tree->root->candbeam->top()->tgt_wids,true);
}

/**************************************************************************************
 1. 函数功能: 为每个句法树节点生成kbest候选
 2. 入口参数: 指向句法树节点的指针
 3. 出口参数: 无
 4. 算法简介: 见注释
************************************************************************************* */
void SentenceTranslator::generate_kbest_for_node(SyntaxNode* node)
{
	Candpq candpq;			//优先级队列, 用来缓存当前句法节点的翻译候选

	vector<RuleMatchInfo> rule_match_info_vec = ruletable->find_matched_rules_for_syntax_node(node);
	//对于匹配上的每条规则, 取出每个非终结符对应的最好候选, 将生成的候选加入candpq
	for (auto &rule_match_info : rule_match_info_vec)
	{
		// 有规则可用, 并且规则源端的头节点与叶子节点不重合(非一元规则)
		if (rule_match_info.rule_node->tgt_rules.size() != 0 && rule_match_info.syntax_leaves.at(0) != rule_match_info.syntax_root)
		{
			add_best_cand_to_pq_for_each_rule(candpq,rule_match_info);
		}
	}

	//TODO
	set<vector<int> > duplicate_set;	//用来记录candpq中的候选是否已经被扩展过
	duplicate_set.clear();
	//立方体剪枝,每次从candpq中取出最好的候选加入当前节点的candbeam中,并将该候选的邻居加入candpq中
	int added_cand_num = 0;				//从candpq中添加进当前候选列表中的候选数
	while(added_cand_num < para.BEAM_SIZE)
	{
		if (candpq.empty()==true)
			break;
		Cand* best_cand = candpq.top();
		candpq.pop();
		if (node->span_lbound == 0 && node->span_rbound == src_sen_len -1)
		{
			double increased_lm_prob = lm_model->cal_final_increased_lm_score(best_cand);
			best_cand->lm_prob += increased_lm_prob;
			best_cand->score += feature_weight.lm*increased_lm_prob;
		}
		bool flag = node->cand_organizer.add(best_cand);
		
		vector<int> key; //TODO
		if ( duplicate_set.find(key) == duplicate_set.end() )
		{
			add_neighbours_to_pq(best_cand,candpq);
			duplicate_set.insert(key);
		}
		if (flag == false)					//如果被丢弃或者替换掉了原来的候选
		{
			delete best_cand;
		}
		else
		{
			added_cand_num++;
		}
	}
	while ( !candpq.empty() )
	{
		delete candpq.top();
		candpq.pop();
	}
}

/**************************************************************************************
 1. 函数功能: 对每个规则匹配信息, 生成候选并加入candpq中
 2. 入口参数: 规则匹配信息, 包括规则Trie节点, 目标端根节点, 目标端叶节点
 3. 出口参数: 缓存当前节点翻译候选的candpq
 4. 算法简介: 见注释
************************************************************************************* */
void SentenceTranslator::add_best_cand_to_pq_for_each_rule(Candpq &candpq, RuleMatchInfo &rule_match_info)
{
	bool is_lexical_rule = true;
	for (auto const &syntax_leaf : rule_match_info.syntax_leaves)  //遍历句法树片段叶节点
	{
		if ( !syntax_leaf->children.empty() )
		{
			is_lexical_rule = false;
			break;
		}
	}

	if (is_lexical_rule == true)    // 词汇化规则, 即规则源端叶节点全是终结符
	{
		for (const auto tgt_rule : rule_match_info.rule_node->tgt_rules)
		{
			Cand* cand = new Cand;
			cand->type = 2;
			cand->tgt_root = tgt_rule.tgt_root;
			cand->lm_prob = lm_model->cal_increased_lm_score(cand);
			cand->score = tgt_rule.score + feature_weight.lm*cand->lm_prob + feature_weight.len*cand->tgt_word_num + feature_weight.compose*tgt_rule.is_composed_rule + feature_weight.derive_len*1;
			candpq.push(cand);
		}
	}
	else // 规则源端叶节点含有非终结符
	{
		vector<map<int, vector<Cand*> >* > cand_group_vec;   // 存储所有非终结符叶节点的候选分组表
		vector<vector<Cand*>* > glue_cands_vec;              // 存储所有非终结符叶节点的glue候选
		for (const auto &syntax_leaf : rule_match_info.syntax_leaves)
		{
			if (syntax_leaf->children.size() == 0)           // 若为词汇节点, 则跳过, 只有非词汇叶节点的翻译候选才被用来生成当前根节点的候选
				continue;
			cand_group_vec.push_back( &(syntax_leaf->cand_organizer.tgt_root_to_cand_group) );
			if (syntax_leaf->cand_organizer.glue_cands.size() == 0)
			{
				glue_cands_vec.push_back(NULL);
			}
			else
			{
				glue_cands_vec.push_back( &(syntax_leaf->cand_organizer.glue_cands) );
			}
		}

		if (rule_match_info.rule_node->proc_flag == false)
		{
			rule_match_info.rule_node->group_and_sort_tgt_rules();                     // 根据规则目标端的非终结符及其对齐关系, 对它们进行分组
		}

		for (auto &kvp : rule_match_info.rule_node->tgt_rule_group)                    // 遍历每一组规则
		{
			TgtRule &best_tgt_rule = kvp.second[0];                                    // 取出每组规则中最好的
			vector<vector<Cand*>* > cands_of_leaves;                                   // 存储规则源端非终结符叶节点的翻译候选
			vector<int> rank_vec;
			Cand* cand = NULL;
			for (size_t i=0;i<best_tgt_rule.aligned_src_positions.size();i++)          // 遍历规则目标端的每一个非终结符叶节点
			{
				int src_idx = best_tgt_rule.aligned_src_positions[i];                  // 该非终结符在规则源端对应的位置
				if (src_idx == -1)
					continue;
				auto it = cand_group_vec[src_idx]->find(best_tgt_rule.tgt_leaves[i]);  // 查找是否有能够匹配该非终结符的翻译候选
				if ( it == cand_group_vec[src_idx]->end() )
				{
					if (glue_cands_vec.size() != 0)                                    // 没有的话就使用glue规则
					{
						cands_of_leaves.push_back(glue_cands_vec[src_idx]);
					}
					else
						goto unmatch;
				}
				else
				{
					cands_of_leaves.push_back( &(it->second) );
				}
			}
			if (best_tgt_rule.tgt_leaves.size() == 0) continue;  //TODO, 不可能为0吧
			
			rank_vec.resize(cands_of_leaves.size(),0);
			cand = generate_cand_from_rule(kvp.second,0,cands_of_leaves,rank_vec);    // 根据规则和叶节点候选生成当前节点的候选
			cand->type = 2;
			candpq.push(cand);
unmatch:;
		}
	}
}

Cand* SentenceTranslator::generate_cand_from_rule(vector<TgtRule> tgt_rules,int rule_rank,vector<vector<Cand*>* > cands_of_leaves, vector<int> cand_rank_vec)
{
	Cand *cand = new Cand;
	//TODO
	return cand;
}

/**************************************************************************************
 1. 函数功能: 合并两个子候选并将生成的候选加入candpq中
 2. 入口参数: 两个子候选,两个子候选的排名
 3. 出口参数: 更新后的candpq
 4. 算法简介: 顺序以及逆序合并两个子候选
************************************************************************************* */
void SentenceTranslator::merge_subcands_and_add_to_pq(Cand* cand_lhs, Cand* cand_rhs,int rank_lhs,int rank_rhs,Candpq &candpq)
{
	
	//TODO
}

/**************************************************************************************
 1. 函数功能: 将当前候选的邻居加入candpq中
 2. 入口参数: 当前候选
 3. 出口参数: 更新后的candpq
 4. 算法简介: //TODO
************************************************************************************* */
void SentenceTranslator::add_neighbours_to_pq(Cand* cur_cand, Candpq &candpq)
{
	//TODO
}
