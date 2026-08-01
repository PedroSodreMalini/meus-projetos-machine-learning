# Bagging - Classificação de Conversão de Lead

## Resumo
Algoritmo de predição se um lead será convertido em venda. O algoritmo usa o modelo `Bagging Classifier`.

## Dataset
|Coluna|Tipo|Significado|
|:-:|:-:|:--|
|Prospect ID|str|ID do Lead|
|Lead Number|int|Número ID do Lead|
|Lead Origin|str|De onde o lead veio(ex: landing page, whatsapp link, etc)|
|Lead Source|str|Plataforma que o lead veio(ex: Google, Tráfego Pago, etc)|
|Do Not Email|str|Indica se o cliente não quer ser respondido por email('Yes' \| 'No')|
|Do Not Call|str|INdica se o cliente não quer ser respondido por ligação('Yes'\|'No')|
|Converted|int|**Variável target** - indica se o lead foi convertido em venda|
|TotalVisits|float|Indica o total de visitas ao conteúdo de venda|
|Total Time Spent on Website|int|Indica o tempo total gasto pelo cliente no website|
|Page Views Per Visit|int|Indica quantas páginas o cliente viu no website|
|Last Activity|str|Indica quando foi a última vez que o cliente acessou o site|
|Country|str|De que país o cliente acessou a anding page|
|Specialization|str|Indica qual serviço o cliente está buscando|
|How did you hear about X Education|str|Como o cliente ouviu falar da empresa 'X Education'|
|What is your current occupation|str|Cargo atual do cliente|
|What matters most to you in choosing a course|str|O que mais importa para o cliente ao escolher um curso|
|Search|str|Indica se o lead veio por pesquisa na web|
|Magazine|str|Indica se o lead veio por uma propaganda em revista|
|Newspaper Article|str|Indica se o lead veio por um artigo em jornal|
|X Education Forums|str|Indica se o lead veio por foruns da empresa|
|Newspaper|str|Indica se o lead veio por uma propaganda em jornal|
|Digital Advertisement|str|Indica se o lead veio por propaganda digital|
|Through Recommendations|str|Indica se o lead veeio por recomendação|
|Receive More Updates About Our Courses|str|Indica se o lead quer obter updates sobre os cursos da empresa|
|Tags|str|Tags que indicam em qual etapa de tracking o lead está|
|Lead Quality|str|Qualidade do Lead|
|Update me on Supply Chain Content|str|Indica se o lead quer ser informado sobre conteúdos de linha de produção|
|Get updates on DM Content|str|Indica se o lead quer ser informado sobre conteúdos em chat privado|
|Lead Profile|str|Perfil do lead|
|City|str|Indica a cidade do Lead|
|Asymmetrique Activity Index|str|Métrica de atividade do lead|
|Asymmetrique Profile Index|str|Métrica de perfil do lead|
|Asymmetrique Activity Score|float|Métrica de atividade do lead|
|Asymmetrique Profile Score|float|Métrica de qualidade do perfil|
|I agree to pay the amount through cheque|str|Indica se o lead acceitar pagar por cheque|
|A free copy of Mastering The Interview|str|Indica se o lead quer uma cópia gratuita de um conteúdo ofertado pela empresa|
|Last Notable Activity|str|Indica a última interação notável do lead|

