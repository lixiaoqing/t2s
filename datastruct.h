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
	int tgt_word_num;			//当前候选目标端的单词数
	vector<int> tgt_wids;		//当前候选目标端的id序列

	//打分信息
	double score;				//当前候选的总得分
	vector<double> trans_probs;	//翻译概率
	double lm_prob;
	double mono_reorder_prob;
	double swap_reorder_prob;

	//合并信息,记录当前候选是由两个子候选合并得来时的相关信息
	int mid;					//记录两个子候选在源语言中的交界位置
	int rank_lhs;				//记录第一个子候选在优先级队列中的排名
	int rank_rhs;				//记录第二个子候选在优先级队列中的排名
	Cand* child_lhs;			//指向第一个子候选的指针
	Cand* child_rhs;			//指向第一个子候选的指针

	//语言模型状态信息
	lm::ngram::ChartState lm_state;

	Cand ()
	{
		beg = 0;
		end = 0;
		phrase_num = 1;

		tgt_word_num = 1;
		tgt_wids.clear();

		score = 0.0;
		trans_probs.clear();
		lm_prob = 0.0;
		mono_reorder_prob = 0.0;
		swap_reorder_prob = 0.0;

		mid = -1;
		rank_lhs = 0;
		rank_rhs = 0;

		child_lhs = NULL;
		child_rhs = NULL;
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

//将跨度相同的候选组织到列表中
class CandBeam
{
	public:
		bool add(Cand *cand_ptr);
		Cand* top() { return data.front(); }
		Cand* at(size_t i) { return data.at(i);}
		int size() { return data.size();  }
		void sort() { std::sort(data.begin(),data.end(),larger); }
	private:
		bool is_bound_same(const Cand *a, const Cand *b);

	private:
		vector<Cand*> data;
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
	string reorder_model_file;
};

struct Parameter
{
	size_t BEAM_SIZE;					//优先级队列的大小限制
	size_t SEN_THREAD_NUM;				//句子级并行数
	size_t SPAN_THREAD_NUM;				//span级并行数
	size_t NBEST_NUM;
	size_t REORDER_WINDOW;     			//最大调序范围
	size_t RULE_NUM_LIMIT;		      	//源端相同的情况下最多能加载的规则数
	bool PRINT_NBEST;
	bool DUMP_RULE;						//是否输出所使用的规则
	bool LOAD_ALIGNMENT;				//加载短语表时是否加载短语内部的词对齐
};

struct Weight
{
	vector<double> trans;
	double lm;
	double reorder_mono;
	double reorder_swap;
	double len;							//译文的单词数
	double phrase_num;					//源端被切成的短语数
};

#endif
