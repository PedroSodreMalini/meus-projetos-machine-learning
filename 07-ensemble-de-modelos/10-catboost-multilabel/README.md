# CatBoost Multilabel

## Resumo
Modelo de classificação multi label de empresas quanto à risco de crédito, de compliance e de mercado.

## Dataset
|Coluna|Tipo|Significado|
|:-:|:-:|:--|
|ID|int|ID único da empresa no dataset|
|Nome_Empresa|str|Nome da empresa|
|Receita_Anual|int|Receita anual da empresa em dólares|
|Margem_Liquida|float|Margem líquida da empresa|
|Endividamento|float|Dívida da empresa|
|Setor|str|Setor da empresa|
|Regiao|str|Continente da Empresa|
|Tempo_Operacao|int|Anos de idade da empresa|
|Auditoria_Externa|int|Se já passou ou não por auditoria externa|
|Rating_Credito|float|Taxa de crédito|
|Tipo_Empresa|str|MEI, Limitada, Multinacional ou S.A|
|Politica_Sustentabilidade|str|Política de sustentabilidade Alta, Média ou Baixa|
|Estrategia_Expansao|str|Estratégia de expansão orgânica, por parceria ou por aquisições|
|Gestao_Risco|str|Gestão de risco centralizada ou descentralizada|
|Cobertura_Seguros|str|Cobertura ampla, básica ou nenhuma|
|Maturidade_Digital|str|Maturidade digital avançada, inicial ou intermediária|
|Governanca_Corporativa|str|Governança corporativa fraca, média ou alta|
|Cultura_Inovacao|str|Cultura de inovação conservadora, neutra ou inovadora|
|Relacao_Comunidade|str|Relação de comunidade ruim, regular, boa ou excelente|
|Risco_Credito|int|**Target** - Risco de crédito (0\|1)|
|Risco_Compliance|int|**Target** - Risco de compliance (0\|1)|
|Risco_Mercado|int|**Target** - Risco de Mercado (0\|1)|
