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
	CandBeam* candbeam = src_tree->root->candbeam;
	for (size_t i=0;i< (candbeam->size()<para.NBEST_NUM?candbeam->size():para.NBEST_NUM);i++)
	{
		TuneInfo tune_info;
		tune_info.sen_id = sen_id;
		tune_info.translation = words_to_str(candbeam->at(i)->tgt_wids,false);
		for (size_t j=0;j<PROB_NUM;j++)
		{
			tune_info.feature_values.push_back(candbeam->at(i)->trans_probs.at(j));
		}
		tune_info.feature_values.push_back(candbeam->at(i)->lm_prob);
		tune_info.feature_values.push_back(candbeam->at(i)->tgt_word_num);
		tune_info.feature_values.push_back(candbeam->at(i)->phrase_num);
		tune_info.total_score = candbeam->at(i)->score;
		nbest_tune_info.push_back(tune_info);
	}
	return nbest_tune_info;
}

vector<string> SentenceTranslator::get_applied_rules(size_t sen_id)
{
	vector<string> applied_rules;
	Cand *best_cand =  src_tree->root->candbeam->top();
	dump_rules(applied_rules,best_cand);
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
	if (cand->child_lhs == NULL)			//左子候选为空时右子候选也为空
	{
		string applied_rule;
		for (size_t i=cand->beg; i<=cand->end; i++)
		{
			applied_rule += src_tree->words.at(i)+" ";
		}
		applied_rule += "||| ";
		for (auto tgt_wid : cand->tgt_wids)
		{
			applied_rule += tgt_vocab->get_word(tgt_wid)+" ";
		}
		TrimLine(applied_rule);
		applied_rules.push_back(applied_rule);
	}
	else
	{
		if (cand->child_lhs->beg < cand->child_rhs->beg)		//子候选为顺序合并
		{
			dump_rules(applied_rules,cand->child_lhs);
			dump_rules(applied_rules,cand->child_rhs);
		}
		else
		{
			dump_rules(applied_rules,cand->child_rhs);
			dump_rules(applied_rules,cand->child_lhs);
		}
	}
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
			if (it == src_tree->nodes_at_span.end())
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
	return words_to_str(src_tree->root->candbeam->top()->tgt_wids,true);
}

/**************************************************************************************
 1. 函数功能: 为每个句法树节点生成kbest候选
 2. 入口参数: 指向句法树节点的指针
 3. 出口参数: 无
 4. 算法简介: 见注释
************************************************************************************* */
void SentenceTranslator::generate_kbest_for_node(SyntaxNode* node)
{
	Candpq candpq_merge;			//优先级队列,用来临时存储通过合并得到的候选

	vector<MatchedRuleStruct> matched_rule_vec = ruletable->find_matched_rules_for_syntax_node(node);
	//对于匹配上的每条规则, 取出每个非终结符对应的最好候选, 将合并得到的候选加入candpq_merge
	//TODO

	set<vector<int> > duplicate_set;	//用来记录candpq_merge中的候选是否已经被扩展过
	duplicate_set.clear();
	//立方体剪枝,每次从candpq_merge中取出最好的候选加入当前节点的candbeam中,并将该候选的邻居加入candpq_merge中
	int added_cand_num = 0;				//从candpq_merge中添加进当前候选列表中的候选数
	while(added_cand_num < para.BEAM_SIZE)
	{
		if (candpq_merge.empty()==true)
			break;
		Cand* best_cand = candpq_merge.top();
		candpq_merge.pop();
		if (node->span_lbound == 0 && node->span_rbound == src_sen_len -1)
		{
			double increased_lm_prob = lm_model->cal_final_increased_lm_score(best_cand);
			best_cand->lm_prob += increased_lm_prob;
			best_cand->score += feature_weight.lm*increased_lm_prob;
		}
		bool flag = node->candbeam->add(best_cand);
		
		vector<int> key; //TODO
		if (duplicate_set.find(key) == duplicate_set.end())
		{
			add_neighbours_to_pq(best_cand,candpq_merge);
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
	while(!candpq_merge.empty())
	{
		delete candpq_merge.top();
		candpq_merge.pop();
	}
}

/**************************************************************************************
 1. 函数功能: 合并两个子候选并将生成的候选加入candpq_merge中
 2. 入口参数: 两个子候选,两个子候选的排名
 3. 出口参数: 更新后的candpq_merge
 4. 算法简介: 顺序以及逆序合并两个子候选
************************************************************************************* */
void SentenceTranslator::merge_subcands_and_add_to_pq(Cand* cand_lhs, Cand* cand_rhs,int rank_lhs,int rank_rhs,Candpq &candpq_merge)
{
	
	Cand* cand_mono = new Cand;
	cand_mono->beg = cand_lhs->beg;
	cand_mono->end = cand_rhs->end;
	cand_mono->mid = cand_rhs->beg;
	cand_mono->tgt_word_num = cand_lhs->tgt_word_num + cand_rhs->tgt_word_num;
	cand_mono->phrase_num = cand_lhs->phrase_num + cand_rhs->phrase_num;
	cand_mono->rank_lhs = rank_lhs;
	cand_mono->rank_rhs = rank_rhs;
	cand_mono->child_lhs = cand_lhs;
	cand_mono->child_rhs = cand_rhs;
	cand_mono->tgt_wids = cand_lhs->tgt_wids;
	cand_mono->tgt_wids.insert(cand_mono->tgt_wids.end(),cand_rhs->tgt_wids.begin(),cand_rhs->tgt_wids.end());
	for (size_t i=0;i<PROB_NUM;i++)
	{
		cand_mono->trans_probs.push_back(cand_lhs->trans_probs.at(i)+cand_rhs->trans_probs.at(i));
	}
	double increased_lm_prob = lm_model->cal_increased_lm_score(cand_mono);
	cand_mono->lm_prob = cand_lhs->lm_prob + cand_rhs->lm_prob + increased_lm_prob;
	cand_mono->score = cand_lhs->score + cand_rhs->score + feature_weight.lm*increased_lm_prob;
	candpq_merge.push(cand_mono);

}

/**************************************************************************************
 1. 函数功能: 将当前候选的邻居加入candpq_merge中
 2. 入口参数: 当前候选
 3. 出口参数: 更新后的candpq_merge
 4. 算法简介: //TODO
************************************************************************************* */
void SentenceTranslator::add_neighbours_to_pq(Cand* cur_cand, Candpq &candpq_merge)
{
	//TODO
}
