#include "ruletable.h"

void RuleTrieNode::group_and_sort_tgt_rules()
{
	for (auto &tgt_rule : tgt_rules)
	{
		if ( tgt_rule.group_id.empty() )    //TODO, 肯定为空吧
		{
			for (size_t i=0; i<tgt_rule.aligned_src_positions.size(); i++)        // 遍历规则目标端叶节点
			{
				if (tgt_rule.aligned_src_positions[i] == -1)                      // 跳过词汇节点
					continue;
				tgt_rule.group_id.push_back(tgt_rule.tgt_leaves[i]);              // 非终结符
				tgt_rule.group_id.push_back(tgt_rule.aligned_src_positions[i]);   // 非终结符在源端句法树片段叶节点中对应的位置
			}
		}

		auto it = tgt_rule_group.find(tgt_rule.group_id);
		if ( it == tgt_rule_group.end() )
		{
			vector<TgtRule> tgt_rules = {tgt_rule};
			tgt_rule_group.insert( make_pair(tgt_rule.group_id,tgt_rules) );
		}
		else
		{
			it->second.push_back(tgt_rule);
		}
	}

	for (auto &kvp : tgt_rule_group)
	{
		sort(kvp.second.begin(),kvp.second.end());
	}

	proc_flag = true;
}

RuleTable::RuleTable(const size_t size_limit,bool load_alignment,const Weight &i_weight,const string &rule_table_file,Vocab *i_src_vocab, Vocab* i_tgt_vocab)
{
	src_vocab = i_src_vocab;
	tgt_vocab = i_tgt_vocab;
	RULE_NUM_LIMIT=size_limit;
	LOAD_ALIGNMENT = load_alignment;
	weight=i_weight;
	root=new RuleTrieNode;
	load_rule_table(rule_table_file);
}

void RuleTable::load_rule_table(const string &rule_table_file)
{
	ifstream fin(rule_table_file.c_str(),ios::binary);
	if (!fin.is_open())
	{
		cerr<<"cannot open rule table file!\n";
		return;
	}
	short int src_rule_len=0;
	while(fin.read((char*)&src_rule_len,sizeof(short int)))
	{
		// 读取规则源端节点的id序列, 规则源端节点为规则源端句法树片段的一层, 存储为规则Trie树中的一个节点
		vector<int> rulenode_ids;
		rulenode_ids.resize(src_rule_len);
		fin.read((char*)&rulenode_ids[0],sizeof(int)*src_rule_len);

		TgtRule tgt_rule;
		//树到树的规则使用, 临时用一下
		fin.read((char*)&tgt_rule.tgt_root,sizeof(int));

		// 读取规则目标端的叶节点序列以及非终结符的对齐
		short int tgt_rule_len;
		fin.read((char*)&tgt_rule_len,sizeof(short int));
		tgt_rule.tgt_leaves.resize(tgt_rule_len);
		fin.read((char*)&(tgt_rule.tgt_leaves[0]),sizeof(int)*tgt_rule_len);
		tgt_rule.aligned_src_positions.resize(tgt_rule_len);
		fin.read((char*)&(tgt_rule.aligned_src_positions[0]),sizeof(int)*tgt_rule_len);
		
		// 规则目标端的词汇个数
		tgt_rule.word_num = 0;
		for (const auto pos : tgt_rule.aligned_src_positions)
		{
			if (pos == -1)
				tgt_rule.word_num++;
		}

		// 规则的6个翻译概率
		tgt_rule.probs.resize(PROB_NUM);
		fin.read((char*)&(tgt_rule.probs[0]),sizeof(double)*PROB_NUM);

		// 规则类型
		fin.read((char*)&tgt_rule.is_composed_rule,sizeof(short int));
		fin.read((char*)&tgt_rule.is_lexical_rule,sizeof(short int));

		if (false)
		{
			for (auto id : rulenode_ids)
			{
				cout<<src_vocab->get_word(id)<<endl;
			}
			cout<<"@@@"<<endl;
			cout<<tgt_vocab->get_word(tgt_rule.tgt_root)<<endl;
			for (auto id : tgt_rule.tgt_leaves)
			{
				cout<<tgt_vocab->get_word(id)<<' ';
			}
			for (auto pos : tgt_rule.aligned_src_positions)
			{
				cout<<pos<<' ';
			}
			cout<<endl;
			for (auto prob : tgt_rule.probs)
			{
				cout<<prob<<' ';
			}
			cout<<endl;
			cout<<tgt_rule.is_composed_rule<<' '<<tgt_rule.is_lexical_rule<<' '<<endl<<endl<<endl;
		}

		if (LOAD_ALIGNMENT == true)
		{
			short int alignment_num=0;
			fin.read((char*)&alignment_num,sizeof(short int));
			int *alignment_array = new int[alignment_num];
			fin.read((char*)alignment_array,sizeof(int)*alignment_num);

			tgt_rule.s2t_pos_map.resize(src_rule_len);
			for(size_t i=0;i<alignment_num/2;i++)
			{
				int ch_pos = alignment_array[2*i];
				int en_pos = alignment_array[2*i+1];
				tgt_rule.s2t_pos_map[ch_pos].push_back(en_pos);
			}
		}


		tgt_rule.score = 0;
		if( tgt_rule.probs.size() != weight.trans.size() )
		{
			cout<<"number of probability in rule is wrong!"<<endl;
		}
		for( size_t i=0; i<weight.trans.size(); i++ )
		{
			tgt_rule.score += tgt_rule.probs[i]*weight.trans[i];
		}

		add_rule_to_trie(rulenode_ids,tgt_rule);
	}
	fin.close();
	cout<<"load rule table file "<<rule_table_file<<" over\n";
}

void RuleTable::add_rule_to_trie(const vector<int> &rulenode_ids, const TgtRule &tgt_rule)
{
	RuleTrieNode* current = root;
	for (const auto &node_id : rulenode_ids)
	{        
		string node_str = src_vocab->get_word(node_id);
		auto it = current->id2subtrie_map.find(node_str);
		if ( it != current->id2subtrie_map.end() )
		{
			current = it->second;
		}
		else
		{
			RuleTrieNode* tmp = new RuleTrieNode();
			current->id2subtrie_map.insert(make_pair(node_str,tmp));
			current = tmp;
		}
	}
	if (current->tgt_rules.size() < RULE_NUM_LIMIT)
	{
		current->tgt_rules.push_back(tgt_rule);
	}
	else
	{
		auto it = min_element(current->tgt_rules.begin(), current->tgt_rules.end());
		if( it->score < tgt_rule.score )
		{
			(*it) = tgt_rule;
		}
	}
}

/**************************************************************************************
 1. 函数功能: 获取当前句法节点匹配到的所有规则
 2. 入口参数: 当前句法节点的指针
 3. 出口参数: 所有匹配上的规则
 4. 算法简介: 按层遍历规则Trie树, 对于每一层中的每一条匹配上的规则, 考察它的下一层规则
************************************************************************************* */
vector<MatchedRuleStruct> RuleTable::find_matched_rules_for_syntax_node(SyntaxNode* cur_node)
{
	vector<MatchedRuleStruct> matched_rule_vec;
	auto it = root->id2subtrie_map.find(cur_node->label);
	if ( it == root->id2subtrie_map.end() )
		return matched_rule_vec;
	MatchedRuleStruct root_rule = {it->second,cur_node,{cur_node}};
	matched_rule_vec.push_back(root_rule);
	size_t last_beg = 0, last_end = matched_rule_vec.size();           // 记录规则Trie树当前层匹配上的规则在matched_rule_vec中的起始和终止位置
	while(last_beg != last_end)
	{
		for (size_t cur_pos=last_beg; cur_pos!=last_end; cur_pos++)    // 对当前层匹配上的规则进行扩展
		{
			push_matched_rules_at_next_level(matched_rule_vec,matched_rule_vec.at(cur_pos));
		}
		last_beg = last_end;
		last_end = matched_rule_vec.size();
	}
	return matched_rule_vec;
}

/**************************************************************************************
 1. 函数功能: 考察当前匹配上的规则的下一层规则, 将能匹配上的加入matched_rule_vec
 2. 入口参数: 当前匹配上的规则的引用
 3. 出口参数: 记录所有匹配规则的matched_rule_vec
 4. 算法简介: 遍历下一层规则, 将每条规则的源端与句法树节点进行对比, 看能否匹配上
************************************************************************************* */
void RuleTable::push_matched_rules_at_next_level(vector<MatchedRuleStruct> &matched_rule_vec, MatchedRuleStruct &cur_rule)
{
	for (auto it=cur_rule.rule_node->id2subtrie_map.begin(); it!=cur_rule.rule_node->id2subtrie_map.end(); it++)
	{
		vector<string> nodes_vec = Split(it->first,"|||");                     // 记录当前规则源端每个叶节点扩展出来的节点
		if (nodes_vec.size() != cur_rule.syntax_leaves.size()) continue;       // TODO: if不可能为true
		vector<SyntaxNode*> new_leaves;
		MatchedRuleStruct new_rule;
		for(size_t i=0; i<cur_rule.syntax_leaves.size(); i++)
		{
			if(nodes_vec[i] == "~")                                            // 该节点不进行扩展, 直接将原规则的叶节点作为新规则的叶节点
			{
				new_leaves.push_back(cur_rule.syntax_leaves[i]);
				continue;
			}
			vector<string> nodes = Split(nodes_vec[i]);                        // 记录规则源端一个叶节点扩展出来的节点
			if( nodes.size() != cur_rule.syntax_leaves[i]->children.size() )   // 规则源端叶节点与对应的句法树叶节点扩展出来的节点数不同
				goto unmatch;
			for(int j=0; j<nodes.size(); j++)                                  // 对规则源端叶节点与对应的句法树叶节点扩展出来的节点进行匹配
			{
				if (nodes[j] != cur_rule.syntax_leaves[i]->children[j]->label)
					goto unmatch;
			}
			new_leaves.insert(new_leaves.end(), cur_rule.syntax_leaves[i]->children.begin(), cur_rule.syntax_leaves[i]->children.end());
		}
		new_rule = {it->second,cur_rule.syntax_root,new_leaves};
		matched_rule_vec.push_back(new_rule);
unmatch:;
	}
}

