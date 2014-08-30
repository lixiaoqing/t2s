#ifndef SYNTAXTREE_H
#define SYNTAXTREE_H
#include "stdafx.h"
#include "myutils.h"
#include "datastruct.h"
#include "vocab.h"

// 源端句法树节点
struct SyntaxNode
{
	string label;                                    // 该节点的句法标签或者词
	SyntaxNode* father;
	vector<SyntaxNode*> children;
	int span_lbound;                                 // 该节点对应的span的左边界
	int span_rbound;                                 // 该节点对应的span的右边界
	CandBeam candbeam_normal;                        // 该节点的普通翻译候选
	CandBeam candbeam_glue;                          // 该节点的glue翻译候选
	map<int,vector<Cand> > tgt_root_to_cand_group;   // 将该节点的翻译候选按照目标端的根节点进行分组
	map<string,int> recombine_info_to_cand_idx;      // 根据重组信息(由边界词与目标端根节点组成)找候选在candbeam_normal中的序号
	                                                 // 以查找新候选是否跟已有候选重复, 如有重复则进行假设重组
	//vector<MatchedRule> matched_rules;               // 该节点匹配到的规则
	
	SyntaxNode ()
	{
		father      = NULL;
		span_lbound = 9999;
		span_rbound = -1;
	}
	
};

class SyntaxTree
{
	public:
		SyntaxTree(const string &line_of_tree);

	private:
		void build_tree_from_str(const string &line_of_tree);
		void update_span(SyntaxNode* node);
		void dump(SyntaxNode* node);

	public:
		SyntaxNode* root;
		int sen_len;
		vector<string> words;
		map<int,vector<SyntaxNode*> > nodes_at_span;    // 记录每个跨度对应的所有节点
};

#endif
