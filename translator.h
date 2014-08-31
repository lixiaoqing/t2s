#include "stdafx.h"
#include "cand.h"
#include "vocab.h"
#include "ruletable.h"
#include "syntaxtree.h"
#include "lm.h"
#include "myutils.h"

struct Models
{
	Vocab *src_vocab;
	Vocab *tgt_vocab;
	RuleTable *ruletable;
	LanguageModel *lm_model;
};

// 记录规则匹配信息, 包括规则Trie树的节点, 以及句法树片段的头节点和叶子节点等信息
struct RuleMatchInfo
{
	RuleTrieNode* rule_node;
	SyntaxNode* syntax_root;
	vector<SyntaxNode*> syntax_leaves;
};

class SentenceTranslator
{
	public:
		SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen);
		~SentenceTranslator();
		string translate_sentence();
		vector<TuneInfo> get_tune_info(size_t sen_id);
		vector<string> get_applied_rules(size_t sen_id);
	private:
		void generate_kbest_for_node(SyntaxNode* node);
		void add_best_cand_to_pq_for_each_rule(Candpq &candpq, RuleMatchInfo &rule_match_info);
		void merge_subcands_and_add_to_pq(Cand *cand_lhs, Cand *cand_rhs,int rank_lhs,int rank_rhs,Candpq &new_cands_by_mergence);
		void add_neighbours_to_pq(Cand *cur_cand, Candpq &new_cands_by_mergence);
		Cand* generate_cand_from_rule(vector<TgtRule> &tgt_rules,int rule_rank,vector<vector<Cand*>* > &cands_of_leaves,vector<int> &cand_rank_vec);
		void dump_rules(vector<string> &applied_rules, Cand *cand);
		string words_to_str(vector<int> wids, bool drop_unk);

		vector<RuleMatchInfo> find_matched_rules_for_syntax_node(SyntaxNode* cur_node);
		void push_matched_rules_at_next_level(vector<RuleMatchInfo> &match_info_vec, RuleMatchInfo &cur_match_info);

	private:
		Vocab *src_vocab;
		Vocab *tgt_vocab;
		RuleTable *ruletable;
		LanguageModel *lm_model;
		Parameter para;
		Weight feature_weight;

		SyntaxTree* src_tree;
		size_t src_sen_len;
};
