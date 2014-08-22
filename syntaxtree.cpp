#include "syntaxtree.h"

SyntaxTree::str_to_tree(string &line_of_tree)
{
	vector<string> toks;
	Split(toks,line_of_tree);

	TreeNode* cur_node;
	TreeNode* pre_node;
	int word_index = 0;
	for(size_t i=0;i<toks.size();i++)
	{
		//左括号情形，且去除"("作为终结符的特例(做终结符时，后继为“）”)
		if(toks[i]=="(" && i+1<toks.size() && toks[i+1]!=")")
		{
			string test=toks[i];
			if(i == 0)
			{
				m_root   = new TreeNode;
				pre_node = m_root;
			}
			else
			{
				cur_node = new TreeNode;
				cur_node->father = pre_node;
				pre_node->children.push_back(cur_node);
				pre_node = cur_node;
			}
		}
		//右括号情形，去除右括号“）”做终结符的特例（做终结符时，前驱的前驱为“（,而且前驱不是")"
		else if(toks[i]==")" && !(i-2>=0 && toks[i-2] =="(" && toks[i-1] != ")"))
		{
			pre_node = pre_node->father;
			cur_node = pre_node;
		}
		//处理形如 （ VV 需要 ）其中VV节点这样的情形
		else if((i-1>=0 && toks[i-1]=="(") && (i+2<toks.size() && toks[i+2]==")"))
		{
			cur_node->label  = toks[i];
			cur_node         = new TreeNode;
			cur_node->father = pre_node;
			pre_node->children.push_back(cur_node);
			word_index++;
		}
		//处理形如 VP （ VV 需要 ） VP这样的节点 或 需要 这样的节点
		else
		{
			cur_node->label = toks[i];
			//如果是“需要”的情形，则记录中文词的序号
			if(toks[i+1]==")")
				cur_node->idx = word_index;
		}
	}
	sen_len = word_index;
}
