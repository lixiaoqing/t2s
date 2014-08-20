#include "stdafx.h"
#include "datastruct.h"

struct TgtRule
{
	bool operator<(const TgtRule &rhs) const{return score<rhs.score;};
	int word_num;                               // 规则目标端的单词数
	vector<int> wids;                           // 规则目标端的单词或非终结符的id序列
	vector<int> aligned_src_positions;          // 规则目标端的单词或非终结符在规则源端句法树片段叶节点序列中的位置, 单词对应-1
	vector<vector<int> > s2t_pos_map;           // 记录规则源端每个单词对应到目标端哪些单词
	double score;                               // 规则打分, 即翻译概率与特征权重的加权
	vector<double> probs;                       // 翻译概率和词汇权重
	bool is_composed_rule;                      //记录该规则是最小规则还是组合规则
	bool is_lexical_rule;                       //记录该规则是完全词汇化规则还是非词汇化规则
};

struct RuleTrieNode 
{
	vector<TgtRule> tgt_rules;                       // 一个规则源端对应的所有目标端
	map <string, vector<TgtRule> > tgt_rulegroup;    // 根据规则目标端的对齐对规则进行分组; TODO: 是否需要
	map <int, RuleTrieNode*> id2subtrie_map;         // 当前规则节点到下个规则节点的转换表; TODO: 是否应该用string做索引
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
		vector<vector<TgtRule>* > find_matched_rules_for_treenode(const TreeNode* cur_node);

	private:
		void load_rule_table(const string &rule_table_file);
		void add_rule_to_trie(const vector<int> &node_ids, const TgtRule &tgt_rule);

	private:
		int RULE_NUM_LIMIT;                      // 每个规则源端最多加载的目标端个数 
		bool LOAD_ALIGNMENT;                     // 是否加载词对齐信息
		RuleTrieNode *root;                      // 规则Trie树根节点
		Weight weight;                           // 特征权重
};
