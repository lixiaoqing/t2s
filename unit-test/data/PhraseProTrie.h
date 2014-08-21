/*
	PhraseProTrie.h
	phrase������ʵĴ洢�ṹ
*/
#ifndef PHRASEPROTRIE_H
#define PHRASEPROTRIE_H
#include "StdAfx.h"
#include "Vocab.h"
#include <vector>
#include <string>

using namespace std;

//�洢Ӣ�Ķ���
class s_PhrasePro 
{	
public:
	bool operator<(const s_PhrasePro &right) const;
	s_PhrasePro& operator=(const s_PhrasePro& right);
	s_PhrasePro()
	{
		ulEnNum = 0;
		viEnPhrase.clear();
		NodeIndexVec.clear();
		composedFlag = 0;
		LexFlag = 0;
		rootIndex = -1;
		dPro = 0.0;
		eachTransPro.clear();
		ngramPro = 0.0;
		strRule = "";
		SrcRule.clear();
	}
	int ulEnNum;						//Ӣ�Ĵʵĸ���

	//��Ҷ�ڵ����У���viEnPhrase��NodeIndexVec��ɵĹ����ʶ��
	string strRule;   
	vector<int> NodeIndexVec;
	vector<int> viEnPhrase;  //����û�б����ķ��������ôӢ�ľͷ��ڸýṹ��
	int composedFlag;     //��¼�ù�������С��������Ϲ���minimal or composed��
	int LexFlag;          //��¼�ù�������ȫ�ʻ㻯�����ǷǴʻ㻯����
	int rootIndex;                //����ĸ��ڵ�
	double dPro;					//����

   //���﷭����ֻ��4�����ʣ�s2tree����5�����ʣ���һ������Ĺ�Լ���ʣ�, t2t�м�������
	vector<double> eachTransPro;
	double ngramPro;           //���ù��������ģ�ͼ��������

	vector<int> SrcRule;   //��¼Դ�˹���
};

class PhraseProTrie
{
public:
	PhraseProTrie()
	{
		_phrasePro.clear();
		_subPhraseTrie.clear();
		_phraseProMap.clear();
		m_ProcFlag = 0;
	//	CandsNum = 0;
	}

	//���ؽڵ��ֵ
	/*vector<vector<s_PhrasePro> > &value()
	{
		return _phrasePro;
	};*/
	vector<s_PhrasePro> &value()
	{
		return _phrasePro;
	};
	map<string, vector<s_PhrasePro> > & map_value()
	{
		return _phraseProMap;
	};

	map<string, PhraseProTrie> & SubTrie()
	{
		return _subPhraseTrie;
	}
	//����һ���ڵ�
	PhraseProTrie* InsertTrie(const string &Index, PhraseProTrie &newTrie)
	{
		_subPhraseTrie.insert(map<string, PhraseProTrie>::value_type(Index,newTrie) );
		return &_subPhraseTrie[Index];
	};
	
	//����һ���ڵ�;
	PhraseProTrie* findTrie(const string &Index)         
	{
		map<string ,PhraseProTrie>::iterator it= _subPhraseTrie.find(Index);
		if( _subPhraseTrie.end() != it )
			return &( (*it).second );
		else
			return NULL;
	};

	PhraseProTrie* matchIndices(vector<string> &Indices);

//	PhraseProTrie* findTrie(const int *Indices);      //����һ�����ڵ�,û�������
	PhraseProTrie* findTrie(vector<string> & Indices); //����һ�����ڵ�,û�������

	//��_phrasePro�е���Ϣ��Ŀ���Ҷ�ڵ����_phraseProMap��,���������
	bool RuleProc();
private:

	//�洢һ��Դ�����Ӧ������Ŀ�����
	map<string, vector<s_PhrasePro> > _phraseProMap;
	vector<s_PhrasePro> _phrasePro;

	//�洢�ڵ�, ����tree2tree����Ҫƥ��Դ�����ṹ����˲����ôʺ���Ϊ������
	map<string, PhraseProTrie> _subPhraseTrie;
public:
	int m_ProcFlag;                     //��ֵ��־����ʾ��Ӧ�Ĺ����Ƿ��Ѵ����
//	int CandsNum;
};

#endif //!define PHRASEPRO_H

