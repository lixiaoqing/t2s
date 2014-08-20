#include "translator.h"

SentenceTranslator::SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen)
{
	src_vocab = i_models.src_vocab;
	tgt_vocab = i_models.tgt_vocab;
	ruletable = i_models.ruletable;
	reorder_model = i_models.reorder_model;
	lm_model = i_models.lm_model;
	para = i_para;
	feature_weight = i_weight;

	stringstream ss(input_sen);
	string word;
	while(ss>>word)
	{
		src_wids.push_back(src_vocab->get_id(word));
	}

	src_sen_len = src_wids.size();
	candbeam_matrix.resize(src_sen_len);
	for (size_t beg=0;beg<src_sen_len;beg++)
	{
		candbeam_matrix.at(beg).resize(src_sen_len-beg);
	}

	fill_matrix_with_matched_rules();
}

SentenceTranslator::~SentenceTranslator()
{
	for (size_t i=0;i<candbeam_matrix.size();i++)
	{
		for(size_t j=0;j<candbeam_matrix.at(i).size();j++)
		{
			for(size_t k=0;k<candbeam_matrix.at(i).at(j).size();k++)
			{
				delete candbeam_matrix.at(i).at(j).at(k);
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 根据短语表中匹配到的所有规则生成翻译候选, 并加入到candbeam_matrix中
 2. 入口参数: 无
 3. 出口参数: 无
 4. 算法简介: a) 如果某个跨度没匹配到规则
              a.1) 如果该跨度包含1个单词, 则生成对应的OOV候选
              a.2) 如果该跨度包含多个单词, 则不作处理
              b) 如果某个跨度匹配到了规则, 则根据规则生成候选
************************************************************************************* */
void SentenceTranslator::fill_matrix_with_matched_rules()
{
	for (size_t beg=0;beg<src_sen_len;beg++)
	{
		vector<vector<TgtRule>* > matched_rules_for_prefixes = ruletable->find_matched_rules_for_prefixes(src_wids,beg);
		for (size_t span=0;span<matched_rules_for_prefixes.size();span++)	//span=0对应跨度包含1个词的情况
		{
			if (matched_rules_for_prefixes.at(span) == NULL)
			{
				if (span == 0)
				{
					Cand* cand = new Cand;
					cand->beg = beg;
					cand->end = beg+span;
					cand->tgt_wids.push_back(tgt_vocab->get_id("NULL"));
					cand->trans_probs.resize(PROB_NUM,LogP_PseudoZero);
					for (size_t i=0;i<PROB_NUM;i++)
					{
						cand->score += feature_weight.trans.at(i)*cand->trans_probs.at(i);
					}
					cand->lm_prob = lm_model->cal_increased_lm_score(cand);
					cand->score += feature_weight.phrase_num*cand->phrase_num 
						       + feature_weight.len*cand->tgt_word_num + feature_weight.lm*cand->lm_prob;
					candbeam_matrix.at(beg).at(span).add(cand);
				}
				continue;
			}
			for (const auto &tgt_rule : *matched_rules_for_prefixes.at(span))
			{
				Cand* cand = new Cand;
				cand->beg = beg;
				cand->end = beg+span;
				cand->tgt_word_num = tgt_rule.word_num;
				cand->tgt_wids = tgt_rule.wids;
				cand->trans_probs = tgt_rule.probs;
				cand->score = tgt_rule.score;
				cand->lm_prob = lm_model->cal_increased_lm_score(cand);
				cand->score += feature_weight.phrase_num*cand->phrase_num 
					       + feature_weight.len*cand->tgt_word_num + feature_weight.lm*cand->lm_prob;
				candbeam_matrix.at(beg).at(span).add(cand);
			}
		}
	}
}

/**************************************************************************************
 1. 函数功能: 计算调序模型打分
 2. 入口参数: 生成当前候选的左候选以及右候选,左右是指在当前候选中的位置
 3. 出口参数: 顺序以及逆序调序概率
 4. 算法简介: 使用如下8个特征估计概率
              {左候选, 右候选} X {源端, 目标端} X {首词, 尾词}
************************************************************************************* */
pair<double,double> SentenceTranslator::cal_reorder_score(const Cand* cand_lhs,const Cand* cand_rhs)
{
	int src_pos_beg_lhs = cand_lhs->beg;
	int src_pos_end_lhs = cand_lhs->end;
	int src_pos_beg_rhs = cand_rhs->beg;
	int src_pos_end_rhs = cand_rhs->end;
	int tgt_wid_beg_lhs = cand_lhs->tgt_wids.at(0);
	int tgt_wid_end_lhs = cand_lhs->tgt_wids.at(cand_lhs->tgt_wids.size()-1);
	int tgt_wid_beg_rhs = cand_rhs->tgt_wids.at(0);
	int tgt_wid_end_rhs = cand_rhs->tgt_wids.at(cand_rhs->tgt_wids.size()-1);
	vector<string> feature_vec;
	feature_vec.resize(8);
	feature_vec.at(0) = "c11=" + src_vocab->get_word(src_wids.at(src_pos_beg_lhs));
	feature_vec.at(1) = "c12=" + src_vocab->get_word(src_wids.at(src_pos_end_lhs));
	feature_vec.at(2) = "c21=" + src_vocab->get_word(src_wids.at(src_pos_beg_rhs));
	feature_vec.at(3) = "c22=" + src_vocab->get_word(src_wids.at(src_pos_end_rhs));
	feature_vec.at(4) = "e11=" + tgt_vocab->get_word(tgt_wid_beg_lhs);
	feature_vec.at(5) = "e12=" + tgt_vocab->get_word(tgt_wid_end_lhs);
	feature_vec.at(6) = "e21=" + tgt_vocab->get_word(tgt_wid_beg_rhs);
	feature_vec.at(7) = "e22=" + tgt_vocab->get_word(tgt_wid_end_rhs);
	vector<double> reorder_prob_vec;
	reorder_model->eval_all(reorder_prob_vec,feature_vec);
	return make_pair(reorder_prob_vec[reorder_model->get_tagid("straight")],reorder_prob_vec[reorder_model->get_tagid("inverted")]);
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
	CandBeam &candbeam = candbeam_matrix.at(0).at(src_sen_len-1);
	for (size_t i=0;i< (candbeam.size()<para.NBEST_NUM?candbeam.size():para.NBEST_NUM);i++)
	{
		TuneInfo tune_info;
		tune_info.sen_id = sen_id;
		tune_info.translation = words_to_str(candbeam.at(i)->tgt_wids,false);
		for (size_t j=0;j<PROB_NUM;j++)
		{
			tune_info.feature_values.push_back(candbeam.at(i)->trans_probs.at(j));
		}
		tune_info.feature_values.push_back(candbeam.at(i)->lm_prob);
		tune_info.feature_values.push_back(candbeam.at(i)->mono_reorder_prob);
		tune_info.feature_values.push_back(candbeam.at(i)->swap_reorder_prob);
		tune_info.feature_values.push_back(candbeam.at(i)->tgt_word_num);
		tune_info.feature_values.push_back(candbeam.at(i)->phrase_num);
		tune_info.total_score = candbeam.at(i)->score;
		nbest_tune_info.push_back(tune_info);
	}
	return nbest_tune_info;
}

vector<string> SentenceTranslator::get_applied_rules(size_t sen_id)
{
	vector<string> applied_rules;
	Cand *best_cand = candbeam_matrix.at(0).at(src_sen_len-1).top();
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
			applied_rule += src_vocab->get_word(src_wids.at(i))+" ";
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
	for(size_t beg=0;beg<src_sen_len;beg++)
	{
		candbeam_matrix.at(beg).at(0).sort();		//对列表中的候选进行排序
	}
	for (size_t span=1;span<src_sen_len;span++)
	{
#pragma omp parallel for num_threads(para.SPAN_THREAD_NUM)
		for(size_t beg=0;beg<src_sen_len-span;beg++)
		{
			generate_kbest_for_span(beg,span);
			candbeam_matrix.at(beg).at(span).sort();
		}
	}
	return words_to_str(candbeam_matrix.at(0).at(src_sen_len-1).top()->tgt_wids,true);
}

/**************************************************************************************
 1. 函数功能: 为每个跨度生成kbest候选
 2. 入口参数: 跨度的起始位置以及跨度的长度(实际为长度减1)
 3. 出口参数: 无
 4. 算法简介: 见注释
************************************************************************************* */
void SentenceTranslator::generate_kbest_for_span(const size_t beg,const size_t span)
{
	Candpq candpq_merge;			//优先级队列,用来临时存储通过合并得到的候选

	//对于当前跨度的每种分割方式,取出左跨度和右跨度中的最好候选,将合并得到的候选加入candpq_merge
	for(size_t span_lhs=0;span_lhs<span;span_lhs++)
	{
		Cand *best_cand_lhs = candbeam_matrix.at(beg).at(span_lhs).top();
		Cand *best_cand_rhs = candbeam_matrix.at(beg+span_lhs+1).at(span-span_lhs-1).top();
		merge_subcands_and_add_to_pq(best_cand_lhs,best_cand_rhs,1,1,candpq_merge);
	}

	set<vector<int> > duplicate_set;	//用来记录candpq_merge中的候选是否已经被扩展过
	duplicate_set.clear();
	//立方体剪枝,每次从candpq_merge中取出最好的候选加入candbeam_matrix中,并将该候选的邻居加入candpq_merge中
	int added_cand_num = 0;				//从candpq_merge中添加进当前候选列表中的候选数
	while(added_cand_num < para.BEAM_SIZE)
	{
		if (candpq_merge.empty()==true)
			break;
		Cand* best_cand = candpq_merge.top();
		candpq_merge.pop();
		if (span == src_sen_len-1)
		{
			double increased_lm_prob = lm_model->cal_final_increased_lm_score(best_cand);
			best_cand->lm_prob += increased_lm_prob;
			best_cand->score += feature_weight.lm*increased_lm_prob;
		}
		bool flag = candbeam_matrix.at(beg).at(span).add(best_cand);
		
		vector<int> key = {best_cand->rank_lhs,best_cand->rank_rhs,best_cand->mid};
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
	double mono_reorder_prob = 0;
	double swap_reorder_prob = 0;
	if (cand_rhs->end - cand_lhs->beg < para.REORDER_WINDOW)
	{
		pair<double,double> reorder_probs = cal_reorder_score(cand_lhs,cand_rhs);
		mono_reorder_prob = reorder_probs.first;
		swap_reorder_prob = reorder_probs.second;
	}
	
	Cand* cand_mono = new Cand;
	cand_mono->beg = cand_lhs->beg;
	cand_mono->end = cand_rhs->end;
	cand_mono->mid = cand_rhs->beg;
	cand_mono->tgt_word_num = cand_lhs->tgt_word_num + cand_rhs->tgt_word_num;
	cand_mono->phrase_num = cand_lhs->phrase_num + cand_rhs->phrase_num;
	cand_mono->mono_reorder_prob = cand_lhs->mono_reorder_prob + cand_rhs->mono_reorder_prob + mono_reorder_prob;
	cand_mono->swap_reorder_prob = cand_lhs->swap_reorder_prob + cand_rhs->swap_reorder_prob;
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
	cand_mono->score = cand_lhs->score + cand_rhs->score 
		           + feature_weight.lm*increased_lm_prob + feature_weight.reorder_mono*mono_reorder_prob;
	candpq_merge.push(cand_mono);

	if (cand_rhs->end - cand_lhs->beg >= para.REORDER_WINDOW)
		return;
	Cand* cand_swap = new Cand;
	*cand_swap = *cand_mono;
	cand_swap->child_lhs = cand_rhs;
	cand_swap->child_rhs = cand_lhs;
	cand_swap->tgt_wids = cand_rhs->tgt_wids;
	cand_swap->tgt_wids.insert(cand_swap->tgt_wids.end(),cand_lhs->tgt_wids.begin(),cand_lhs->tgt_wids.end());
	cand_swap->mono_reorder_prob = cand_lhs->mono_reorder_prob + cand_rhs->mono_reorder_prob;
	cand_swap->swap_reorder_prob = cand_lhs->swap_reorder_prob + cand_rhs->swap_reorder_prob + swap_reorder_prob;
	increased_lm_prob = lm_model->cal_increased_lm_score(cand_swap);
	cand_swap->lm_prob = cand_lhs->lm_prob + cand_rhs->lm_prob + increased_lm_prob;
	cand_swap->score = cand_lhs->score + cand_rhs->score 
		           + feature_weight.lm*increased_lm_prob + feature_weight.reorder_mono*swap_reorder_prob;
	candpq_merge.push(cand_swap);
}

/**************************************************************************************
 1. 函数功能: 将当前候选的邻居加入candpq_merge中
 2. 入口参数: 当前候选
 3. 出口参数: 更新后的candpq_merge
 4. 算法简介: a) 取比当前候选左子候选差一名的候选与当前候选的右子候选合并
              b) 取比当前候选右子候选差一名的候选与当前候选的左子候选合并
************************************************************************************* */
void SentenceTranslator::add_neighbours_to_pq(Cand* cur_cand, Candpq &candpq_merge)
{
	int beg = cur_cand->beg;
	int end = cur_cand->end;
	int mid = cur_cand->mid;
	int span_lhs = mid - beg - 1;
	int span_rhs = end - mid;

	int rank_lhs = cur_cand->rank_lhs + 1;
	int rank_rhs = cur_cand->rank_rhs;
	if(candbeam_matrix.at(beg).at(span_lhs).size() >= rank_lhs)
	{
		Cand *cand_lhs = candbeam_matrix.at(beg).at(span_lhs).at(rank_lhs-1);
		Cand *cand_rhs = candbeam_matrix.at(mid).at(span_rhs).at(rank_rhs-1);
		merge_subcands_and_add_to_pq(cand_lhs,cand_rhs,rank_lhs,rank_rhs,candpq_merge);
	}

	rank_lhs = cur_cand->rank_lhs;
	rank_rhs = cur_cand->rank_rhs + 1;
	if(candbeam_matrix.at(mid).at(span_rhs).size() >= rank_rhs)
	{
		Cand *cand_lhs = candbeam_matrix.at(beg).at(span_lhs).at(rank_lhs-1);
		Cand *cand_rhs = candbeam_matrix.at(mid).at(span_rhs).at(rank_rhs-1);
		merge_subcands_and_add_to_pq(cand_lhs,cand_rhs,rank_lhs,rank_rhs,candpq_merge);
	}
}
