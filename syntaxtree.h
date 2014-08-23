#ifndef SYNTAXTREE_H
#define SYNTAXTREE_H
#include "stdafx.h"
#include "myutils.h"
#include "datastruct.h"

// 源端句法树节点
struct TreeNode
{
	string label;                                    // 该节点的句法标签或者词
	TreeNode* father;
	vector<TreeNode*> children;
	int span_lbound;                                 // 该节点对应的span的左边界
	int span_rbound;                                 // 该节点对应的span的右边界
	CandBeam candbeam;                               // 该节点的翻译候选
	//vector<MatchedRule> matched_rules;               // 该节点匹配到的规则
	
	TreeNode ()
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
		void update_span(TreeNode* node);
		void dump(TreeNode* node);

	public:
		TreeNode* root;
		int sen_len;
		vector<string> words;
		map<int,vector<TreeNode*> > span_to_node;    // 记录每个跨度对应的所有节点
};

#endif
