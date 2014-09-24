#ifndef CAND_H
#define CAND_H
#include "stdafx.h"
#include "ruletable.h"
#include "lm/left.hh"

//存储翻译候选
struct Cand	                
{
	//目标端信息
	vector<int> tgt_wids;		//当前候选译文的单词id序列

	//打分信息
	double score;				//当前候选的总得分
	vector<double> trans_probs;	//翻译概率
	double lm_prob;

	//来源信息, 记录候选是如何生成的
	CandType type;                                 // 候选的类型(1.由OOV生成; 2.由普通规则生成; 3.由glue规则生成)
	string syntax_node_info;                       // 当前候选所对应的句法节点信息(包括句法标签和跨度), 输出规则信息时用
	RuleTrieNode* rule_node;                       // 生成当前候选的规则的源端
	vector<TgtRule>* matched_tgt_rules;            // 目标端非终结符个数及对齐相同的一组规则
	int rule_rank;                                 // 当前候选所用的规则在matched_tgt_rules中的排名
	vector<vector<Cand*> > cands_of_nt_leaves;     // 规则源端非终结符叶节点的翻译候选(glue规则所有叶节点均为非终结符)
	vector<int> cand_rank_vec;                     // 记录当前候选所用的每个非终结符叶节点的翻译候选的排名
	int rule_num;                                  // 使用的规则的数量

	//语言模型状态信息
	lm::ngram::ChartState lm_state;

	Cand ()
	{
		tgt_wids.clear();

		score = 0.0;
		trans_probs.resize(PROB_NUM,0);
		lm_prob = 0.0;

		type = INIT;
		rule_node = NULL;
		matched_tgt_rules = NULL;
		rule_rank = 0;
		cands_of_nt_leaves.clear();
		cand_rank_vec.clear();
		rule_num  = 0;
	}
	~Cand ()
	{
		tgt_wids.resize(0);

		score = 0.0;
		trans_probs.resize(0);
		lm_prob = 0.0;

		type = INIT;
		rule_node = NULL;
		matched_tgt_rules = NULL;
		rule_rank = 0;
		cands_of_nt_leaves.resize(0);
		cand_rank_vec.resize(0);
		rule_num  = 0;
	}
};

struct smaller
{
	bool operator() ( const Cand *pl, const Cand *pr )
	{
		return pl->score < pr->score;
	}
};

bool larger( const Cand *pl, const Cand *pr );

//组织每个句法节点翻译候选的类
class CandOrganizer
{
	public:
		~CandOrganizer()
		{
			for (auto cand : all_cands)
			{
				delete cand;
			}
			for (auto cand : recombined_cands)
			{
				delete cand;
			}
		}
		bool add(Cand *&cand_ptr);
		void sort() {std::sort(all_cands.begin(),all_cands.end(),larger);};
	
	private:
		bool is_bound_same(const Cand *a, const Cand *b);

	public:
		vector<Cand*> all_cands;                         // 当前节点所有的翻译候选
		vector<Cand*> recombined_cands;                  // 被重组的翻译候选, 回溯查看所用规则的时候使用
};

typedef priority_queue<Cand*, vector<Cand*>, smaller> Candpq;

#endif
