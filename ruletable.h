#include "stdafx.h"
#include "datastruct.h"

struct TgtRule
{
	bool operator<(const TgtRule &rhs) const{return score<rhs.score;};
	int word_num;                               // 规则目标端的单词数
	vector<int> wids;                           // 规则目标端的单词id序列
	double score;                               // 规则打分, 即翻译概率与特征权重的加权
	vector<double> probs;                       // 翻译概率和词汇权重
	vector<vector<int> > s2t_pos_map;           // 记录每个源端位置对应到哪些目标端位置
};

struct RuleTrieNode 
{
	vector<TgtRule> tgt_rules;                  // 一个规则源端对应的所有目标端
	map <int, RuleTrieNode*> id2subtrie_map;    // 当前规则节点到下个规则节点的转换表
};

class RuleTable
{
	public:
		RuleTable(const size_t size_limit,bool load_alignment,const Weight &i_weight,const string &rule_table_file)
		{
			RULE_NUM_LIMIT=size_limit;
			LOAD_ALIGNMENT = load_alignment;
			weight=i_weight;
			root=new RuleTrieNode;
			load_rule_table(rule_table_file);
		};
		vector<vector<TgtRule>* > find_matched_rules_for_prefixes(const vector<int> &src_wids,const size_t pos);

	private:
		void load_rule_table(const string &rule_table_file);
		void add_rule_to_trie(const vector<int> &src_wids, const TgtRule &tgt_rule);

	private:
		int RULE_NUM_LIMIT;                      // 每个规则源端最多加载的目标端个数 
		bool LOAD_ALIGNMENT;                     // 是否加载词对齐信息
		RuleTrieNode *root;                      // 规则Trie树根节点
		Weight weight;                           // 特征权重
};
