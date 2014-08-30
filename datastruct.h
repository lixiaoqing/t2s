#ifndef DATASTRUCT_H
#define DATASTRUCT_H
#include "stdafx.h"
#include "lm/left.hh"

//存储翻译候选
struct Cand	                
{
	//源端信息
	int beg;					//当前候选在源语言中的起始位置
	int end;					//当前候选在源语言中的终止位置
	int phrase_num;				//当前候选包含的短语数

	//目标端信息
	int tgt_root;               //当前候选目标端的根节点
	int tgt_word_num;			//当前候选目标端的单词数
	vector<int> tgt_wids;		//当前候选目标端的id序列

	//打分信息
	double score;				//当前候选的总得分
	vector<double> trans_probs;	//翻译概率
	double lm_prob;

	//生成信息, 记录候选是如何生成的
	int type;                   //候选的类型(0:由OOV生成; 1:由glue规则; 2:由普通规则生成; 3:由一元规则生成)

	//语言模型状态信息
	lm::ngram::ChartState lm_state;

	Cand ()
	{
		beg = 0;
		end = 0;
		phrase_num = 1;

		tgt_root = -1;
		tgt_word_num = 1;
		tgt_wids.clear();

		score = 0.0;
		trans_probs.clear();
		lm_prob = 0.0;

		type = 0;
	}
};

struct smaller
{
	bool operator() ( const Cand *pl, const Cand *pr )
	{
		return pl->score < pr->score;
	}
};

bool larger( const Cand *pl, const Cand *pr );

//组织每个句法节点翻译候选的类
class CandOrganizer
{
	public:
		bool add(Cand *cand_ptr);
		Cand* top() { return normal_cands.front(); }
		Cand* at(size_t i) { return normal_cands.at(i);}
		int size() { return normal_cands.size();  }
		void sort() { std::sort(normal_cands.begin(),normal_cands.end(),larger); }
	private:
		bool is_bound_same(const Cand *a, const Cand *b);

	public:
		vector<Cand*> normal_cands;                      // 当前节点的普通翻译候选
		vector<Cand*> glue_cands;                        // 当前节点的glue翻译候选
		map<int,vector<Cand*> > tgt_root_to_cand_group;  // 将当前节点的翻译候选按照目标端的根节点进行分组
		map<string,int> recombine_info_to_cand_idx;      // 根据重组信息(由边界词与目标端根节点组成)找候选在candbeam_normal中的序号
	                                                     // 以查找新候选是否跟已有候选重复, 如有重复则进行假设重组
};

typedef priority_queue<Cand*, vector<Cand*>, smaller> Candpq;

struct TuneInfo
{
	size_t sen_id;
	string translation;
	vector<double> feature_values;
	double total_score;
};

struct Filenames
{
	string input_file;
	string output_file;
	string nbest_file;
	string src_vocab_file;
	string tgt_vocab_file;
	string rule_table_file;
	string lm_file;
};

struct Parameter
{
	size_t BEAM_SIZE;					//优先级队列的大小限制
	size_t SEN_THREAD_NUM;				//句子级并行数
	size_t SPAN_THREAD_NUM;				//span级并行数
	size_t NBEST_NUM;
	size_t RULE_NUM_LIMIT;		      	//源端相同的情况下最多能加载的规则数
	bool PRINT_NBEST;
	bool DUMP_RULE;						//是否输出所使用的规则
	bool LOAD_ALIGNMENT;				//加载短语表时是否加载短语内部的词对齐
};

struct Weight
{
	vector<double> trans;
	double lm;
	double len;							//译文的单词数
	double phrase_num;					//源端被切成的短语数
	double compose;
	double derive_len;
};

#endif
