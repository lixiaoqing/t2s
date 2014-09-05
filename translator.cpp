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

string SentenceTranslator::words_to_str(vector<int> &wids, bool drop_unk)
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
***************************************************************************************/
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
				for (const auto &node : it->second)
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
***************************************************************************************/
void SentenceTranslator::generate_kbest_for_node(SyntaxNode* node)
{
	vector<RuleMatchInfo> rule_match_info_vec = find_matched_rules_for_syntax_node(node);  // 查找匹配的规则
	if (rule_match_info_vec.empty() && node->span_lbound==node->span_rbound && node->children.size()==1 && node->children[0]->children.size()==0)
	{
		add_cand_for_oov(node);                                                            // 为OOV生成候选, 并加入当前节点的翻译候选中
	}
	else
	{
		Candpq candpq;			                                                           // 优先级队列, 用来缓存当前句法节点的翻译候选
		for (auto &rule_match_info : rule_match_info_vec)                                  // 遍历匹配上的规则
		{
			if (rule_match_info.rule_node->tgt_rules.size() != 0 && rule_match_info.syntax_leaves.at(0) != rule_match_info.syntax_root)
			{
				add_best_cand_to_pq_with_normal_rule(candpq,rule_match_info);              // 根据(非一元)规则生成候选, 并加入candpq
			}
		}
		if (candpq.size()==0)
		{
			add_best_cand_to_pq_with_glue_rule(candpq,node);                               // 使用glue规则生成候选, 并加入candpq
		}
		extend_cand_by_cube_pruning(candpq,node);                                          // 通过立方体剪枝对候选进行扩展
		if (rule_match_info_vec[0].rule_node->tgt_rules.size() != 0)
		{
			extend_cand_with_unary_rule(rule_match_info_vec[0]);                           // 根据一元规则对候选进行扩展
		}
		node->cand_organizer.sort_and_group_cands();                                       // 对候选进行排序和分组
		while ( !candpq.empty() )
		{
			delete candpq.top();
			candpq.pop();
		}
	}
}

void SentenceTranslator::add_cand_for_oov(SyntaxNode *node)
{
	Cand *oov_cand = new Cand;
	oov_cand->type = 0;
	oov_cand->trans_probs.resize(PROB_NUM,LogP_PseudoZero);
	for (const auto w : feature_weight.trans)
	{
		oov_cand->score += w*LogP_PseudoZero;
	}
	oov_cand->tgt_root     = tgt_vocab->get_id("NN");
	oov_cand->tgt_wids     = {tgt_vocab->get_id("NULL")};
	oov_cand->rule_num     = 1;
	oov_cand->lm_prob      = lm_model->cal_increased_lm_score(oov_cand);
	oov_cand->score       += feature_weight.lm*oov_cand->lm_prob + feature_weight.compose*1 + feature_weight.rule_num*1;
	node->cand_organizer.add(oov_cand);
}

/**************************************************************************************
 1. 函数功能: 对每个规则匹配信息, 生成候选并加入candpq中
 2. 入口参数: 规则匹配信息, 包括规则Trie节点, 目标端根节点, 目标端叶节点
 3. 出口参数: 缓存当前节点翻译候选的candpq
 4. 算法简介: 见注释
***************************************************************************************/
void SentenceTranslator::add_best_cand_to_pq_with_normal_rule(Candpq &candpq, RuleMatchInfo &rule_match_info)
{
	if (rule_match_info.rule_node->proc_flag == false)
	{
		rule_match_info.rule_node->group_and_sort_tgt_rules();                           // 根据规则目标端的非终结符及其对齐关系进行分组
	}

	vector<map<int, vector<Cand*> >* > cand_group_vec;                                   // 存储所有非终结符叶节点的候选分组表
	for (const auto &syntax_leaf : rule_match_info.syntax_leaves)
	{
		if (syntax_leaf->children.size() == 0)                                           // 若为词汇节点, 则跳过
			continue;
		map<int,vector<Cand*> > &cand_group = syntax_leaf->cand_organizer.tgt_root_to_cand_group;
		cand_group_vec.push_back(&cand_group);
	}

	for (auto &kvp : rule_match_info.rule_node->tgt_rule_group)                          // 遍历规则目标端的分组
	{
		TgtRule &best_tgt_rule = kvp.second[0];                                          // 取出每组规则中最好的
		vector<vector<Cand*> > cands_of_nt_leaves;                                       // 存储规则源端非终结符叶节点的翻译候选
		vector<int> rank_vec;
		Cand *cand = NULL;
		for (size_t i=0;i<best_tgt_rule.aligned_src_positions.size();i++)                // 遍历规则目标端的每一个叶节点
		{
			int src_idx = best_tgt_rule.aligned_src_positions[i];                        // 该叶节点在规则源端对应的位置
																						 // TODO src_idx的值使用叶节点序号(此为非终结符序号)更方便
			if (src_idx == -1)                                                           // 跳过终结符叶节点, 若全是终结符则cands_of_nt_leaves为空
				continue;
			auto it = cand_group_vec[src_idx]->find(best_tgt_rule.tgt_leaves[i]);
			auto it_glue = cand_group_vec[src_idx]->find( tgt_vocab->get_id("X-X-X") );
			if ( it != cand_group_vec[src_idx]->end() )                                  // 有能够匹配该非终结符的翻译候选
			{
				cands_of_nt_leaves.push_back(it->second);
			}
			else if ( it_glue != cand_group_vec[src_idx]->end() )                        // 没有匹配候选就使用glue候选 TODO 不应该用吧
			{
				cands_of_nt_leaves.push_back(it_glue->second);
			}
			else
				goto unmatch;
		}
		if (best_tgt_rule.tgt_leaves.size() == 0) continue;                              // TODO, 不可能为0吧
		rank_vec.resize(cands_of_nt_leaves.size(),0);
		cand = generate_cand_from_normal_rule(kvp.second,0,cands_of_nt_leaves,rank_vec); // 根据规则和叶节点候选生成当前节点的候选
		candpq.push(cand);
unmatch:;
	}
}

/**************************************************************************************
 1. 函数功能: 根据规则和非终结符叶节点的翻译候选生成当前节点的候选
 2. 入口参数: a) 非终结符叶节点相同的规则列表 b) 使用的规则在规则列表中的排名
              c) 每个非终结符叶节点的翻译候选 d) 使用的每个翻译候选在它所在列表中的排名
 3. 出口参数: 指向新生成的候选的指针
 4. 算法简介: 见注释
***************************************************************************************/
Cand* SentenceTranslator::generate_cand_from_normal_rule(vector<TgtRule> &tgt_rules,int rule_rank,vector<vector<Cand*> > &cands_of_nt_leaves, vector<int> &cand_rank_vec)
{
	Cand *cand = new Cand;
	cand->type = 1;
	// 记录当前候选的以下来源信息: 1) 使用的哪条规则; 2) 使用的每个非终结符叶节点中的哪个候选; 3) 使用的每个叶节点候选的目标端根节点id
	cand->matched_tgt_rules  = &tgt_rules;
	cand->rule_rank          = rule_rank;
	cand->cands_of_nt_leaves = cands_of_nt_leaves;
	cand->cand_rank_vec      = cand_rank_vec;
	for (size_t i=0; i<cands_of_nt_leaves.size(); i++)
	{
		cand->tgt_root_of_leaf_cands.push_back(cands_of_nt_leaves[i][cand_rank_vec[i]]->tgt_root);
	}
	
	TgtRule &applied_rule = tgt_rules[rule_rank];
	cand->tgt_root        = applied_rule.tgt_root;
	cand->trans_probs     = applied_rule.probs;
	size_t nt_idx = 0;
	for (size_t i=0; i<applied_rule.tgt_leaves.size(); i++)
	{
		if (applied_rule.aligned_src_positions[i] == -1)
		{
			cand->tgt_wids.push_back(applied_rule.tgt_leaves[i]);
		}
		else
		{
			Cand* subcand = cands_of_nt_leaves[nt_idx][cand_rank_vec[nt_idx]];
			cand->tgt_wids.insert(cand->tgt_wids.begin(),subcand->tgt_wids.begin(),subcand->tgt_wids.end());
			cand->rule_num += subcand->rule_num;
			for (size_t j=0; j<PROB_NUM; j++)
			{
				cand->trans_probs[j] += subcand->trans_probs[j];
			}
			cand->lm_prob += subcand->lm_prob;
			cand->score   += subcand->score;
			nt_idx++;
		}
	}
	double increased_lm_score = lm_model->cal_increased_lm_score(cand); 
	cand->rule_num += 1;
	cand->lm_prob  += increased_lm_score;
	cand->score    += applied_rule.score + feature_weight.lm*increased_lm_score + feature_weight.len*applied_rule.word_num
                      + feature_weight.compose*applied_rule.is_composed_rule + feature_weight.rule_num*1;
	return cand;
}

void SentenceTranslator::add_best_cand_to_pq_with_glue_rule(Candpq &candpq,SyntaxNode* node)
{
	vector<vector<Cand*> > cands_of_leaves;                                // 存储规则源端所有叶节点的翻译候选
	for (auto &syntax_leaf : node->children)
	{
		cands_of_leaves.push_back(syntax_leaf->cand_organizer.all_cands);
	}
	vector<int> cand_rank_vec(cands_of_leaves.size(),0);
	Cand *glue_cand = generate_cand_from_glue_rule(cands_of_leaves,cand_rank_vec);
	candpq.push(glue_cand);
}

Cand* SentenceTranslator::generate_cand_from_glue_rule(vector<vector<Cand*> > &cands_of_leaves, vector<int> &cand_rank_vec)
{
	Cand *glue_cand = new Cand;
	glue_cand->type = 2;
	glue_cand->cands_of_nt_leaves = cands_of_leaves;
	glue_cand->cand_rank_vec      = cand_rank_vec;
	glue_cand->tgt_root   = tgt_vocab->get_id("X-X-X");
	for (size_t i=0; i<cands_of_leaves.size(); i++)
	{
		Cand *subcand = cands_of_leaves[i][cand_rank_vec[i]];
		glue_cand->tgt_root_of_leaf_cands.push_back(subcand->tgt_root);
		glue_cand->tgt_wids.insert( glue_cand->tgt_wids.end(),subcand->tgt_wids.begin(),subcand->tgt_wids.end() );
		glue_cand->rule_num += subcand->rule_num;
		for (size_t j=0; j<PROB_NUM; j++)
		{
			glue_cand->trans_probs[j] += subcand->trans_probs[j];
		}
		glue_cand->lm_prob += subcand->lm_prob;
		glue_cand->score   += subcand->score;
	}
	double increased_lm_score = lm_model->cal_increased_lm_score(glue_cand);
	glue_cand->lm_prob   += increased_lm_score;
	glue_cand->rule_num   =  1;
	glue_cand->score += feature_weight.lm*increased_lm_score + feature_weight.glue*1 + feature_weight.rule_num*1;
	return glue_cand;
}

void SentenceTranslator::extend_cand_by_cube_pruning(Candpq &candpq, SyntaxNode* node)
{
	set<vector<int> > duplicate_set;
	for (size_t i=0; i<para.BEAM_SIZE;i++)
	{
		if (candpq.empty())
			break;
		Cand *best_cand = candpq.top();
		candpq.pop();
		node->cand_organizer.add(best_cand);
		add_neighbours_to_pq(candpq,best_cand,duplicate_set);
	}
}

/**************************************************************************************
 1. 函数功能: 将当前候选的邻居加入candpq中
 2. 入口参数: 当前候选, 检查是否重复扩展的duplicate_set
 3. 出口参数: 更新后的candpq
 4. 算法简介: a) 对于glue规则生成的候选, 考虑它所有叶节点的下一位候选
              b) 对于普通规则生成的候选, 考虑叶节点候选的下一位以及规则的下一位
***************************************************************************************/
void SentenceTranslator::add_neighbours_to_pq(Candpq &candpq, Cand* cur_cand, set<vector<int> > &duplicate_set)
{
	vector<int> base_key;
	base_key.push_back(cur_cand->tgt_root);
	base_key.insert(base_key.end(),cur_cand->tgt_root_of_leaf_cands.begin(),cur_cand->tgt_root_of_leaf_cands.end());
    // 遍历所有非终结符叶节点, 若候选所用规则目标端无非终结符则不会进入此循环
	for (size_t i=0; i<cur_cand->cands_of_nt_leaves.size(); i++)
	{
		if ( cur_cand->cand_rank_vec[i]+1 < cur_cand->cands_of_nt_leaves[i].size() )
		{
			vector<int> new_cand_rank_vec = cur_cand->cand_rank_vec;
			new_cand_rank_vec[i]++;
			vector<int> new_key = base_key;
			if (cur_cand->type == 1)
			{
				new_key.push_back(cur_cand->rule_rank);
			}
			new_key.insert( new_key.end(),new_cand_rank_vec.begin(),new_cand_rank_vec.end() );
			if (duplicate_set.count(new_key) != 0)
			{
				Cand *new_cand;
				if (cur_cand->type == 1)    // 普通规则生成的候选
				{
					new_cand = generate_cand_from_normal_rule(*(cur_cand->matched_tgt_rules),cur_cand->rule_rank,cur_cand->cands_of_nt_leaves,new_cand_rank_vec);
				}
				else if (cur_cand->type == 2)         // glue规则生成的候选
				{
					new_cand = generate_cand_from_glue_rule(cur_cand->cands_of_nt_leaves,new_cand_rank_vec);
				}
				candpq.push(new_cand);
				duplicate_set.insert(new_key);
			}
		}
	}
    // 对普通规则生成的候选, 考虑规则的下一位
	if (cur_cand->type == 1 && cur_cand->rule_rank+1<cur_cand->matched_tgt_rules->size())
	{
		vector<int> new_key = base_key;
		new_key.push_back(cur_cand->rule_rank+1);
		new_key.insert( new_key.end(),cur_cand->cand_rank_vec.begin(),cur_cand->cand_rank_vec.end() );
		if (duplicate_set.count(new_key) != 0)
		{
			Cand *new_cand = generate_cand_from_normal_rule(*(cur_cand->matched_tgt_rules),cur_cand->rule_rank,cur_cand->cands_of_nt_leaves,cur_cand->cand_rank_vec);
			candpq.push(new_cand);
			duplicate_set.insert(new_key);
		}
	}
}

void SentenceTranslator::extend_cand_with_unary_rule(RuleMatchInfo &rule_match_info)
{
	if (rule_match_info.rule_node->proc_flag == false)
	{
		rule_match_info.rule_node->group_and_sort_tgt_rules();
	}
	for (auto &cand : rule_match_info.syntax_root->cand_organizer.all_cands)
	{
		if ( cand->tgt_root == tgt_vocab->get_id("X-X-X") )
			continue;
		vector<int> tgt_root_id = {cand->tgt_root,tgt_vocab->get_id("-0@")};  //TODO
		auto it = rule_match_info.rule_node->tgt_rule_group.find(tgt_root_id);
		if ( it == rule_match_info.rule_node->tgt_rule_group.end() )
			continue;
		vector<TgtRule> &tgt_rules = it->second;
		vector<vector<Cand*> > cands_of_nt_leaves = {{cand}};
		vector<int> cand_rank_vec = {0};
		for (size_t rule_rank=0;rule_rank<tgt_rules.size();rule_rank++)
		{
			Cand *new_cand = generate_cand_from_normal_rule(tgt_rules,rule_rank,cands_of_nt_leaves,cand_rank_vec);
			rule_match_info.syntax_root->cand_organizer.add(new_cand);
		}
	}
}

/**************************************************************************************
 1. 函数功能: 获取当前句法节点匹配到的所有规则
 2. 入口参数: 当前句法节点的指针
 3. 出口参数: 所有匹配上的规则
 4. 算法简介: 按层遍历规则Trie树, 对于每一层中的每一条匹配上的规则, 考察它的下一层规则
************************************************************************************* */
vector<RuleMatchInfo> SentenceTranslator::find_matched_rules_for_syntax_node(SyntaxNode* cur_node)
{
	vector<RuleMatchInfo> match_info_vec;
	auto it = ruletable->get_root()->id2subtrie_map.find(cur_node->label);
	if ( it == ruletable->get_root()->id2subtrie_map.end() )
		return match_info_vec;
	RuleMatchInfo root_rule = {it->second,cur_node,{cur_node}};        // 一元规则, 即规则源端只含一个根节点
	match_info_vec.push_back(root_rule);
	size_t last_beg = 0, last_end = match_info_vec.size();             // 记录规则Trie树当前层匹配上的规则在match_info_vec中的起始和终止位置
	while(last_beg != last_end)
	{
		for (size_t cur_pos=last_beg; cur_pos!=last_end; cur_pos++)    // 对当前层匹配上的规则进行扩展
		{
			push_matched_rules_at_next_level(match_info_vec,match_info_vec.at(cur_pos));
		}
		last_beg = last_end;
		last_end = match_info_vec.size();
	}
	return match_info_vec;
}

/**************************************************************************************
 1. 函数功能: 考察当前匹配上的规则的下一层规则, 将能匹配上的加入match_info_vec
 2. 入口参数: 当前匹配上的规则的引用
 3. 出口参数: 记录所有匹配规则的match_info_vec
 4. 算法简介: 遍历下一层规则, 将每条规则的源端与句法树节点进行对比, 看能否匹配上
***************************************************************************************/
void SentenceTranslator::push_matched_rules_at_next_level(vector<RuleMatchInfo> &match_info_vec, RuleMatchInfo &cur_match_info)
{
	for (auto &kvp : cur_match_info.rule_node->id2subtrie_map)
	{
		vector<string> nodes_vec = Split(kvp.first,"|||");                         // 记录当前规则源端每个叶节点扩展出来的节点
		if (nodes_vec.size() != cur_match_info.syntax_leaves.size()) continue;     // TODO: if不可能为true
		vector<SyntaxNode*> new_leaves;
		RuleMatchInfo new_match_info;
		for(size_t i=0; i<cur_match_info.syntax_leaves.size(); i++)
		{
			if(nodes_vec[i] == "~")                                                // 该节点不进行扩展, 直接将原规则的叶节点作为新规则的叶节点
			{
				new_leaves.push_back(cur_match_info.syntax_leaves[i]);
				continue;
			}
			vector<string> nodes = Split(nodes_vec[i]);                            // 记录规则源端一个叶节点扩展出来的节点
			if( nodes.size() != cur_match_info.syntax_leaves[i]->children.size() ) // 规则源端叶节点与对应的句法树叶节点扩展出来的节点数不同
				goto unmatch;
			for(int j=0; j<nodes.size(); j++)                                      // 对规则源端叶节点与对应的句法树叶节点扩展出来的节点进行匹配
			{
				if (nodes[j] != cur_match_info.syntax_leaves[i]->children[j]->label)
					goto unmatch;
			}
			new_leaves.insert(new_leaves.end(), cur_match_info.syntax_leaves[i]->children.begin(), cur_match_info.syntax_leaves[i]->children.end());
		}
		new_match_info = {kvp.second,cur_match_info.syntax_root,new_leaves};
		match_info_vec.push_back(new_match_info);
unmatch:;
	}
}

