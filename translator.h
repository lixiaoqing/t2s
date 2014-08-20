#include "stdafx.h"
#include "datastruct.h"
#include "vocab.h"
#include "ruletable.h"
#include "lm.h"
#include "maxent.h"
#include "myutils.h"

struct Models
{
	Vocab *src_vocab;
	Vocab *tgt_vocab;
	RuleTable *ruletable;
	MaxentModel *reorder_model;
	LanguageModel *lm_model;
};

class SentenceTranslator
{
	public:
		SentenceTranslator(const Models &i_models, const Parameter &i_para, const Weight &i_weight, const string &input_sen);
		~SentenceTranslator();
		string translate_sentence();
		vector<TuneInfo> get_tune_info(size_t sen_id);
		vector<string> get_applied_rules(size_t sen_id);
	private:
		void fill_matrix_with_matched_rules();
		pair<double,double> cal_reorder_score(const Cand *cand_lhs,const Cand *cand_rhs);
		void generate_kbest_for_span(const size_t beg,const size_t span);
		void merge_subcands_and_add_to_pq(Cand *cand_lhs, Cand *cand_rhs,int rank_lhs,int rank_rhs,Candpq &new_cands_by_mergence);
		void add_neighbours_to_pq(Cand *cur_cand, Candpq &new_cands_by_mergence);
		void dump_rules(vector<string> &applied_rules, Cand *cand);
		string words_to_str(vector<int> wids, bool drop_unk);

	private:
		Vocab *src_vocab;
		Vocab *tgt_vocab;
		RuleTable *ruletable;
		MaxentModel *reorder_model;
		LanguageModel *lm_model;
		Parameter para;
		Weight feature_weight;

		vector<vector<CandBeam> > candbeam_matrix;		//存储解码过程中所有跨度对应的候选列表, 
													    //candbeam_matrix[i][j]存储起始位置为i, 跨度为j的候选列表
		vector<int> src_wids;
		size_t src_sen_len;
};
