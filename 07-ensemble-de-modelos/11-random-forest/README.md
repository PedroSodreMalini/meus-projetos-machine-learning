# Random Forest Classification

## Resumo

Identificação de quando um empregado sai da empresa.

## Dataset

|Nome da Coluna|Tipo da Coluna| Significado da coluna|
|:-:|:-:|:--|
|`id` | `string` | Identificador único do funcionário.|
| `idade` | `int64`| Idade do funcionário em anos.|
|`genero` |`category`| Gênero informado pelo funcionário. |
| `estado_civil`|`category`| Estado civil do funcionário. |
| `educacao`|`category`| Nível de escolaridade do funcionário.|
|`regime_trabalho`|`category`| Modalidade de trabalho: presencial, remoto ou híbrido. |
| `data_contratacao`| `datetime64[ns]` | Data em que o funcionário foi contratado pela empresa. |
| `data_demissao` | `datetime64[ns]` / `NaT` | Data de desligamento do funcionário. Nula (`NaT`) enquanto o funcionário permanece na empresa. |
| `tipo_demissao` |`category` / `NaN`| Tipo de desligamento: voluntária ou involuntária. Nulo para funcionários que não tiveram desligamento. |
| `cargo` |`category`| Cargo ocupado pelo funcionário.|
| `salario_atual` |`float64` | Salário atual do funcionário.|
| `data_ultimo_feedback`| `datetime64[ns]` | Data em que o funcionário recebeu seu feedback mais recente. |
|`data_ultimo_aumento`| `datetime64[ns]` | Data em que o funcionário recebeu seu aumento salarial mais recente. |
| `data_ultima_mudanca_cargo` | `datetime64[ns]` | Data da mudança de cargo mais recente do funcionário.|
|`nota_avaliacao` |`float64` | Nota atribuída ao desempenho do funcionário em sua avaliação.|
| `acompanhamento_psicologo`|`bool`| Indica se o funcionário realiza acompanhamento psicológico (`True`/`False`). |
| `qtde_projetos` | `int64`| Quantidade de projetos nos quais o funcionário está ou estava envolvido. |
| `qtde_clientes` | `int64`| Quantidade de clientes associados ao trabalho do funcionário.|
|`nivel_satisfacao_gestor`|`float64` | Nível de satisfação do funcionário em relação ao gestor, provavelmente em uma escala de 0 a 10.|
| `churn` | `int64` / `bool` | Variável target que indica se o funcionário saiu da empresa: `1` para desligamento e `0` para permanência. |
