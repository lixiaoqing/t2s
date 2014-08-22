#include "stdafx.h"
#include "syntaxtree.h"
#include "vocab.h"

struct TgtRule
{
	bool operator<(const TgtRule &rhs) const{return score<rhs.score;};
	int word_num;                               // 规则目标端的单词数
	vector<int> treenode_ids;                   // 规则目标端的单词或非终结符的id序列
	vector<int> aligned_src_positions;          // 规则目标端的单词或非终结符在规则源端句法树片段叶节点序列中的位置, 单词对应-1
	vector<int> group_id;                       // 规则目标端的非终结符及其对齐所构成的规则分组标识符
	vector<vector<int> > s2t_pos_map;           // 记录规则源端每个单词对应到目标端哪些单词
	double score;                               // 规则打分, 即翻译概率与特征权重的加权
	vector<double> probs;                       // 翻译概率和词汇权重
	short int is_composed_rule;                 // 记录该规则是最小规则还是组合规则
	short int is_lexical_rule;                  // 记录该规则是完全词汇化规则还是非词汇化规则
};

struct RuleTrieNode 
{
	vector<TgtRule> tgt_rules;                             // 一个规则源端对应的所有目标端
	map <vector<int>, vector<TgtRule> > tgt_rule_group;    // 根据规则目标端的句法标签对规则进行分组, 对s2t/t2t系统有用
	map <int, RuleTrieNode*> id2subtrie_map;               // 当前规则节点到下个规则节点的转换表
};

class RuleTable
{
	public:
		RuleTable(const size_t size_limit,bool load_alignment,const Weight &i_weight,const string &rule_table_file,Vocab *i_src_vocab, Vocab* i_tgt_vocab)
		{
			src_vocab = i_src_vocab;
			tgt_vocab = i_tgt_vocab;
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
		Vocab *src_vocab;
		Vocab *tgt_vocab;
};
