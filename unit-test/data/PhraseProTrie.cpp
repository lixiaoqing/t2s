/*
PhraseProTrie.cpp
*/


#include "PhraseProTrie.h"

string Int2Str(int a)
{
	ostringstream s;
	s << a;
	return s.str(); //��intת��Ϊstring
}

bool 
s_PhrasePro::operator <(const s_PhrasePro &right)const
{
//	for(int i=0; i<NodeIndexVec.size(); i++)
//		strRule = I2S(viEnPhrase[i])+I2S(NodeIndexVec[i]);
//	for(int i=0; i<right.NodeIndexVec.size(); i++)
//		right.strRule = I2S(right.viEnPhrase[i])+I2S(right.NodeIndexVec[i]);

//	if(strRule == right.strRule)
	return dPro < right.dPro;
//	else
//	return strRule < right.strRule;
}

s_PhrasePro&
s_PhrasePro::operator =(const s_PhrasePro &right)
{
	ulEnNum = right.ulEnNum;					
	viEnPhrase = right.viEnPhrase;
	NodeIndexVec = right.NodeIndexVec;
	strRule = right.strRule;
	SrcRule = right.SrcRule;
	//viLeftofLeftEn = right.viLeftofLeftEn;
	//viRightofLeftEn = right.viRightofLeftEn;
	//viRightofRightEn = right.viRightofRightEn;
	//invertedFlag = right.invertedFlag;
	composedFlag = right.composedFlag;
	LexFlag = right.LexFlag;
	rootIndex = right.rootIndex;
	dPro = right.dPro;						
	eachTransPro = right.eachTransPro;
	ngramPro = right.ngramPro;
	return *this;
}

////////////////////////////////////////////////////////////////////////
// 1.��Ҫ���ܣ����Ҷ��ڵ㣨�������Ķ�����ң�ÿһ��洢һ�����Ĵʣ�
// 2.��ڲ��������Ķ���Ĵʺ�
// 3.���ڲ������ڵ��ָ��
// 4.�������ڣ�2006.04.10
// 5.�㷨��飺
//
/*
PhraseProTrie* 
PhraseProTrie::findTrie(const int *Indices)
{
	int m_len= Vocab::length(Indices);
        //cout<<m_len<<endl;
	PhraseProTrie* subTrie=this;
	for(int i=0; i<m_len; i++)
	{
		PhraseProTrie* m_subTemp=subTrie->findTrie(Indices[i]);
		if( m_subTemp )
			subTrie = m_subTemp;
		else
		{
			PhraseProTrie m_triTemp;
			subTrie=subTrie->InsertTrie(Indices[i],m_triTemp);
		}
	}
	return subTrie;
}
*/

PhraseProTrie * PhraseProTrie::findTrie(vector<string> & Indices)
{
	int m_len= Indices.size();
	//cout<<m_len<<endl;
	PhraseProTrie* subTrie=this;
	for(int i=0; i<m_len; i++)
	{
		PhraseProTrie* m_subTemp=subTrie->findTrie(Indices[i]);
		if( m_subTemp )
			subTrie = m_subTemp;
		else
		{
			PhraseProTrie m_triTemp;
			subTrie=subTrie->InsertTrie(Indices[i],m_triTemp);
		}
	}
	return subTrie;
}

//�ú���ֻ������ң�������������
PhraseProTrie* PhraseProTrie::matchIndices(vector<string> &Indices)
{
	int m_len = Indices.size();
	PhraseProTrie* subTrie = this;
	for( int i = 0; i < m_len; i++ )
	{
		PhraseProTrie* m_subTemp = subTrie->findTrie(Indices[i]);
		if( m_subTemp )
			subTrie = m_subTemp;
		else
			return NULL;
	}
	return subTrie;
}

bool PhraseProTrie::RuleProc()
{
	if(!m_ProcFlag)
	{
		vector<s_PhrasePro>::iterator iter = _phrasePro.begin();
		for(; iter != _phrasePro.end(); ++iter)
		{
/*			for(int j=0; j<iter->NodeIndexVec.size(); j++)
				cout<<(iter->viEnPhrase[j])<<" ";
			cout<<this<<endl;
*/
			//���ɱ���������ַ�����
			//�ַ���������ֻ��Ҷ�ڵ�Tag��Ҳ�����ٰ�����Դ�˵����������ߵõ���ͬmap
			//�˴���ʹ��
			if(iter->strRule == "")
			{
				string strRule = "";
				for(int i=0; i<(iter->NodeIndexVec).size(); i++)
				{
					if(iter->NodeIndexVec[i] == -1)
						continue;
					strRule = strRule + Int2Str((iter->viEnPhrase)[i]) +"-"+Int2Str((iter->NodeIndexVec)[i])+"@";
				}
				iter->strRule = strRule;
			}

			//���뵽Map��
			map<string, vector<s_PhrasePro> >::iterator m_iter = _phraseProMap.find(iter->strRule);
			if(m_iter == _phraseProMap.end())
			{
				vector<s_PhrasePro> c_PhraseProVec;
				c_PhraseProVec.push_back(*iter);
				_phraseProMap.insert(map<string, vector<s_PhrasePro> >::value_type(iter->strRule, c_PhraseProVec));
			}
			else
				m_iter->second.push_back(*iter);
		}

		//��Map�еĹ��򰴴�ֽ�������
		map<string, vector<s_PhrasePro> >::iterator m_iter = _phraseProMap.begin();
		for(; m_iter!=_phraseProMap.end(); ++m_iter)
		{
			sort(m_iter->second.rbegin(), m_iter->second.rend());
		}

		m_ProcFlag = 1;
	}
}

/*
bool PhraseProTrie::UnaryProc()
{
	if(!m_UnaryFlag)
	{
		vector<s_PhrasePro>::iterator iter = _phrasePro.begin();
		for(; iter != _phrasePro.end(); ++iter)
		{
			int VarNum = 0;
			int VarIndex = -1;
			for(int j=0; j<iter->NodeIndexVec.size(); j++)
			{
				if(NodeIndexVec[j] == -1)
				   continue;	
				
				VarNum++;
				if(VarNum > 1)
					break;
				VarIndex = _phrasePro.viEnPhrase[NodeIndexVec[i]];
			}
			if(VarNum != 1)
				continue;

			map<int, vector<s_PhrasePro*> >::iterator m_iter = _UnaryProMap.find(VarIndex);
			if(m_iter == _UnaryProMap.end())
			{
				vector<s_PhrasePro*> c_PhraseProVec;
				c_PhraseProVec.push_back(&(*iter));
				_UnaryProMap.insert(map<int, vector<s_PhrasePro*> >::value_type(int, c_PhraseProVec));
			}
			else
				m_iter->second.push_back(&(*iter));
		}

	//	map<string, vector<s_PhrasePro> >::iterator m_iter = _phraseProMap.begin();
	//	for(; m_iter!=_phraseProMap.end(); ++m_iter)
	//	{
	//		sort(m_iter->second.rbegin(), m_iter->second.rend());
	//	}

	//	_phrasePro.clear();
		m_UnaryFlag = 1;
	}
}*/
