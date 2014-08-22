#include "stdafx.h"
#include "myutils.h"
#include "datastruct.h"

// 源端句法树节点
struct TreeNode
{
	string label;                                    // 该节点的句法标签或者词
	size_t idx;                                      // 该节点在句子中的位置, 只对词汇节点有效
	TreeNode* father;
	vector<TreeNode*> children;
	int span_lbound;                                 // 该节点对应的span的左边界
	int span_rbound;                                 // 该节点对应的span的右边界
	CandBeam candbeam;                               // 该节点的翻译候选
	//vector<MatchedRule> matched_rules;               // 该节点匹配到的规则
	
	TreeNode ()
	{
		idx         = -1;
		father      = NULL;
		span_lbound = -1;
		span_rbound = -1;
	}
	
};

class SyntaxTree
{
	public:
		void str_to_tree(string &line_of_tree);

	private:
		TreeNode* root;
		int sen_len;
		map<int,vector<TreeNode*> > span_to_node;    // 记录每个跨度对应的所有节点
};

