#include "ruletable.h"

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

		//树到树的规则使用, 临时用一下
		int tgt_treefrag_root;
		fin.read((char*)&tgt_treefrag_root,sizeof(int));

		// 读取规则目标端的叶节点序列以及非终结符的对齐
		short int tgt_rule_len;
		fin.read((char*)&tgt_rule_len,sizeof(short int));
		TgtRule tgt_rule;
		tgt_rule.syntaxnode_ids.resize(tgt_rule_len);
		fin.read((char*)&(tgt_rule.syntaxnode_ids[0]),sizeof(int)*tgt_rule_len);
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
			cout<<tgt_vocab->get_word(tgt_treefrag_root)<<endl;
			for (auto id : tgt_rule.syntaxnode_ids)
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

vector<MatchedRuleStruct> RuleTable::find_matched_rules_for_syntax_node(const SyntaxNode* cur_node)
{
	vector<MatchedRuleStruct> matched_rule_vec;
	RuleTrieNode* current = root;
	// TODO
	return matched_rule_vec;
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
