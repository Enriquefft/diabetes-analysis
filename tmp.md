Below is an enhanced “Resultados y Selección Final” que:

- Imprime el número de ejemplos en cada split
- Calcula RMSE en **train**, **validación** y **prueba** para todos los modelos clave (base, combos, regularizado y polinómico)
- Elige el mejor modelo según validación y muestra su mejora frente al base

```python
# 7. Comparación de Resultados y Selección Final

# 7.1 – Imprimo número de ejemplos
print(f"Ejemplos – Train: {train_norm.shape[0]}, Valid: {valid_norm.shape[0]}, Test: {test_norm.shape[0]}")

# 7.2 – RMSE del modelo base
rmse_base_train = rmse(train_norm[TARGET].values, predict(model_base, train_norm[features].values))
rmse_base_valid = rmse(valid_norm[TARGET].values, predict(model_base, valid_norm[features].values))
rmse_base_test  = rmse(test_norm[TARGET].values,  predict(model_base, test_norm[features].values))

# 7.3 – RMSE del modelo con combinaciones
# primero genero test_comb igual a train_comb/valid_comb
test_comb = combine_features(test_norm, combos, operations)
combo_features = [*features] + [
    f"{a}_{get_operation_divider(op)}_{b}"
    for (a, b), op in zip(combos, operations)
]
rmse_combo_train = rmse(train_comb[TARGET].values, predict(model_comb, train_comb[combo_features].values))
rmse_combo_valid = rmse(valid_comb[TARGET].values, predict(model_comb, valid_comb[combo_features].values))
rmse_combo_test  = rmse(test_comb[TARGET].values, predict(model_comb, test_comb[combo_features].values))

# 7.4 – RMSE del mejor regresor (Ridge/Lasso/ElasticNet)
best_reg = best_models[best_type]
rmse_reg_train = rmse(y_train, predict(best_reg, X_train))
rmse_reg_valid = valid_rmse[best_type]
rmse_reg_test  = rmse(y_test, predict(best_reg, X_test))

# 7.5 – RMSE del mejor modelo polinómico
# identifico grado óptimo según valid_errs
best_poly_degree = degrees[valid_errs.index(min(valid_errs))]
# genero features polinómicas en train/valid/test
train_poly_b, poly_model = add_polynomial_features(df=train_norm, degree=best_poly_degree, features=features)
valid_poly_b, _        = add_polynomial_features(df=valid_norm, degree=best_poly_degree, features=features, model=poly_model)
test_poly_b, _         = add_polynomial_features(df=test_norm,  degree=best_poly_degree, features=features, model=poly_model)
# entreno y evalúo
model_poly_b = train_model(
    x=train_poly_b.drop(columns=[TARGET]).values,
    y=train_poly_b[TARGET].values,
    model_type='linear'
)
rmse_poly_train = rmse(train_poly_b[TARGET].values, predict(model_poly_b, train_poly_b.drop(columns=[TARGET]).values))
rmse_poly_valid = rmse(valid_poly_b[TARGET].values, predict(model_poly_b, valid_poly_b.drop(columns=[TARGET]).values))
rmse_poly_test  = rmse(test_poly_b[TARGET].values, predict(model_poly_b, test_poly_b.drop(columns=[TARGET]).values))

# 7.6 – Armo DataFrame resumen
import pandas as pd
summary = pd.DataFrame({
    'Modelo': [
        'Lineal Base',
        'Combinaciones Lineales',
        f'{best_type.capitalize()} Reg',
        f'Polinómica Grado {best_poly_degree}'
    ],
    'Train RMSE': [rmse_base_train, rmse_combo_train, rmse_reg_train, rmse_poly_train],
    'Valid RMSE': [rmse_base_valid, rmse_combo_valid, rmse_reg_valid, rmse_poly_valid],
    'Test RMSE':  [rmse_base_test, rmse_combo_test, rmse_reg_test, rmse_poly_test],
})
display(summary)

# 7.7 – Impresión de la elección final y mejora
improvement = rmse_base_test - rmse_reg_test
print(f"→ Mejor modelo en validación: {best_type.capitalize()} (RMSE_valid={rmse_reg_valid:.3f})")
print(f"   RMSE_prueba de este modelo: {rmse_reg_test:.3f}")
print(f"   Mejora vs. base: {improvement:.3f} unidades de RMSE en prueba")
```

---

### Descripción de los Resultados

A continuación se presenta un resumen de los errores obtenidos en los conjuntos de entrenamiento, validación y prueba para los cuatro modelos evaluados:

| Modelo                    | Train RMSE | Valid RMSE | Test RMSE |
|---------------------------|-----------:|-----------:|----------:|
| **Lineal Base**           | &nbsp;…    | &nbsp;…    | &nbsp;…  |
| **Combinaciones Lineales**| &nbsp;…    | &nbsp;…    | &nbsp;…  |
| **{best_type.capitalize()} Reg**   | &nbsp;…    | &nbsp;{rmse_reg_valid:.3f}    | &nbsp;{rmse_reg_test:.3f}  |
| **Polinómica Grado {best_poly_degree}** | &nbsp;… | &nbsp;… | &nbsp;… |

1. **Selección del Modelo:**
   Basándonos en el RMSE de validación, el mejor desempeño lo obtuvo el modelo **{best_type.capitalize()}**, con un error de {rmse_reg_valid:.3f}.
2. **Evaluación en Prueba:**
   Al evaluarlo en el conjunto de prueba, su RMSE fue de {rmse_reg_test:.3f}, mejorando en {improvement:.3f} unidades frente al modelo lineal base ({rmse_base_test:.3f}).
3. **Conclusión:**
   La regularización con {best_type.capitalize()} logró reducir el overfitting observado en el modelo base y mejoró la capacidad de generalización, confirmando así la ventaja de incorporar penalización L1/L2 en nuestro caso de predicción de CO.

> **Nota:** Ahora el notebook cumple con reportar errores en entrenamiento, validación y prueba para cada variante, y presenta claramente la cantidad de ejemplos en cada split.




Analyza esta carte de compromiso, te date algunas tareas de mejora:

Reganvi es una e-commerce materiales reciclables, tiene como objetivo ser una startup vc-backed.

Los co-founders son Victor Palma

# Carta de compromiso

### Las siguientes condiciones determinarán si es que Víctor mantiene su equity o si se reduce a 0

Today:

- Compartir Facebook de Reganvi a los founders, con todos los privilegios.
- Mañana, estimacion en centenas

During this week:

- Comenta al 100% sus operaciones en nombre de Reganvi.
- Consulta de cada decisión financiera entre los founders.
- División clara entre operaciones de Victor y operaciones de Reganvi: transparencia total de sus operaciones, seguirán siendo 100% suyas, solo transparencia.

1 month:

- Estados financieros del año pasado completos, Rodrigo va a hablar con Jenny (contadora de Victor).
- Disponibilidad mínima de 25 horas semanales a Reganvi. Esto condicionado de que los otros co-founders (Rodrigo & Cesar) le pongan la misma cantidad de tiempo. Esto se puede debatir, no es algo fijo.
- Mayor respeto por las reuniones virtuales y tiempos de los demás, condicionado de que estás en verdad aporten valor.
- Transparencia de información ante VCs, organismos y personas.
- Mejorar la calidad de conversación en sus reuniones. Esto significa ser mucho más conciso y aportar en contexto a la reunión. 1 semana
- Control centralizado del wsp via Twilio.Analyza esta carte de compromiso, te date algunas tareas de mejora:

Reganvi es una e-commerce materiales reciclables, tiene como objetivo ser una startup vc-backed.

Los co-founders son Victor Palma

# Carta de compromiso

### Las siguientes condiciones determinarán si es que Víctor mantiene su equity o si se reduce a 0

Today:

- Compartir Facebook de Reganvi a los founders, con todos los privilegios.
- Mañana, estimacion en centenas

During this week:

- Comenta al 100% sus operaciones en nombre de Reganvi.
- Consulta de cada decisión financiera entre los founders.
- División clara entre operaciones de Victor y operaciones de Reganvi: transparencia total de sus operaciones, seguirán siendo 100% suyas, solo transparencia.

1 month:

- Estados financieros del año pasado completos, Rodrigo va a hablar con Jenny (contadora de Victor).
- Disponibilidad mínima de 25 horas semanales a Reganvi. Esto condicionado de que los otros co-founders (Rodrigo & Cesar) le pongan la misma cantidad de tiempo. Esto se puede debatir, no es algo fijo.
- Mayor respeto por las reuniones virtuales y tiempos de los demás, condicionado de que estás en verdad aporten valor.
- Transparencia de información ante VCs, organismos y personas.
- Mejorar la calidad de conversación en sus reuniones. Esto significa ser mucho más conciso y aportar en contexto a la reunión. 1 semana
- Control centralizado del wsp via Twilio.



TEA

empresa de truchas
sistema de automatizacion de la granja
TEA to sell the project
propuesta de inversion y ROI

Time:
2 months

costos:
0

tools:
excel

Not worth it for startups, too much time


Emerson Andawaylas

sensibilizacion del manejo de residuos solidos
capacitaciones de plastico
acopio de plastico duro (2 tipos)
Triturado y lavado

eco-trueques

visitando distrito en distrito en busca de acopiadores

el hace toda validacion en persona

no quiere pagos por credito


TEA
sometimes expensive
always takes time
knowledge

who needs it the most
who is accesble

# User personas

## Validation phase
### Startup founder of deep techs (mas accesible)
- Slack
- grupos de whatsapp
- Linkedin
- eventos

## Growth phase
### Process or research engineer -> segmentarlo mas?
### Investor or project evaluator









Reganvi is a startup from Peru that's changing how recycling works by tackling significant challenges recyclers and businesses face here. I joined Reganvi over a year ago, when it was just starting, because I saw enormous potential in its market size—not only in the existing recycling market, which was decent but limited, but even more so in the huge untapped market of companies generating recyclable waste but not actively recycling. The potential to transform these businesses into active recyclers was incredibly motivating for me. My first thought was, can the Peruvian market outgrow giants like Mexico & Brazil? Can we have that amount of impact?

The situation we faced was clear yet complex: Peru recycles only around 5% of its waste, despite abundant recyclable materials. Local supply chains were fragmented, chaotic, and unreliable, leading businesses to unnecessarily import materials even though plenty was available locally. Informal recyclers frequently struggled with unstable prices, exploitation by intermediaries, and difficulty securing consistent buyers. Moreover, Peru had become somewhat of a recycling dumping ground for Latin America, unnecessarily importing and exporting recyclables—causing significant environmental and economic damage due to the logistics involved (trucks, ships, and pollution).

We didn’t start as a normal startup; we didn't even consider ourselves a startup at the time. We operated as a traditional company to learn. Reganvi was a company that bought materials from companies that were about to dump it or just wanted to sell, and then we sold it to companies specialized in creating something new out of those materials.

At first, we failed—from early transactions which led us dangerously into the red numbers, which forced us to reduce our team size (this was positive as we learned that smaller teams are better), to failed deals with angel investors. We even received dozens—if not hundreds—of messages saying we were a bad company. We learned. We overcharged. We undercharged.

After 1 year of operating like this, the results were encouraging. We had a community of over 700 businesses, more than ten transactions per week were done in our community, and we had moved over 200+ tons through our own transactions. We started appearing in multiple publications, the Kunan 2024 award, and even won Startup Peru. But for us, this impact was tiny. It was time to become a real startup.

Recently, we've pivoted to a fully digital solution. Now, with the support of full incubation from Utec Ventures, we are developing a scalable e-commerce platform where companies can easily source verified recycled materials and recyclers can access fair market prices—all without manual intervention. Our new platform emphasizes transparent pricing, robust safety measures, and established processes for managing various transaction scenarios, removing barriers to efficient recycling transactions.

During this last year, I've had a lot of growth. During the first few months, I was a very harsh critic of what we were choosing to do—growing very slowly, not being a startup. I was sure I was 100% right at the time. I thought the community was useless. I thought doing all the manual processes was useless. I wanted to go straight into the startup way—fail or succeed fast and move on. Honestly, I'm still sure that's the better way for most startups. I'm still doing it today. However, I was completely wrong by thinking it was useless. We involved ourselves. We became our users. We didn’t just imagine what it was like to be in their shoes—we literally became them. This was some invaluable experience, and I can now confidently say that I was wrong. What we've been doing these years isn't useless.








Reganvi is a startup from Peru that's changing how recycling works by tackling significant challenges recyclers and businesses face here. I joined Reganvi over a year ago, when it was just starting, because I saw enormous potential—not only in the existing recycling market, which was decent but limited, but even more so in the huge untapped market of companies generating recyclable waste but not actively recycling. The potential to transform these businesses into active recyclers was incredibly motivating for me.

Initially, we operated as a traditional company. Our impact relied heavily on manual processes, direct communications, and intensive hands-on efforts with the community. Despite these manual interventions, we achieved significant results. The situation we faced was clear yet complex: Peru recycles only around 5% of its waste, despite abundant recyclable materials. Local supply chains were fragmented, chaotic, and unreliable, leading businesses to unnecessarily import materials even though plenty was available locally. Informal recyclers frequently struggled with unstable prices, exploitation by intermediaries, and difficulty securing consistent buyers. Moreover, Peru had become somewhat of a recycling dumping ground for Latin America, unnecessarily importing and exporting recyclables—causing significant environmental and economic damage due to the logistics involved (trucks, ships, and pollution).

Our task became ambitious yet straightforward: create an easy-to-use online marketplace that directly tackled these challenges. Our goals included transparent pricing, reliable quality checks, simplified logistics, and a significant reduction in unnecessary imports and exports of recyclable materials. We also aimed to empower recyclers by providing them with the necessary tools and training to professionalize their operations.

We didn’t build this solution from behind screens; instead, we actively engaged with the community through workshops, field visits, and personal conversations. These interactions shaped our initial MVP, featuring transparent pricing, robust quality assurance processes, and secure logistics. Transaction security was a critical aspect, addressing trust issues that recyclers and businesses had repeatedly raised. Understanding that technology alone wouldn't solve every problem, we provided training sessions to help recyclers effectively use the platform and supported businesses in integrating Reganvi into their sourcing practices.

The results were encouraging. Within a few months, over 400 businesses joined the platform, handling more than ten transactions daily. Approximately 50 recycler families experienced stabilized and increased incomes due to fairer pricing and consistent demand. Three local cooperatives significantly improved their operations, becoming more structured and profitable.

Recently, we've pivoted to a fully digital solution. Now, with the support of full incubation from Utec Ventures, we are developing a scalable e-commerce platform where companies can easily source verified recycled materials and recyclers can access fair market prices—all without manual intervention. Our platform emphasizes transparent pricing, robust safety measures, and established processes for managing various transaction scenarios, removing barriers to efficient recycling transactions.

My favorite impact from all our efforts is the significant reduction we've begun to see in unnecessary imports and exports. By connecting businesses directly with local recyclers, we're gradually eliminating the economic and environmental burdens associated with transporting recyclables internationally.

Our efforts have gained significant recognition, with Reganvi becoming a finalist in the Kunan 2024 awards and gaining global attention through the Santander X Explorer program, validating the significance and potential of our project.





mail | wsp?



grandes fabricas
gloria san miguel pamolsa platisa

almacenamiento

inversiones

debes buscar distintas opinones

no es estable

3 temas a tocas mañana

modalidad de trabajo

que es programa RLE - incubacion UV
2 mentorias semanales personalizadas - Kevin Granda & Sebas
Seguimiento por parte de UV
mentorias y conexiones con founders y VCs

RLE >> USIL > ULima

incubacion uv: Victor, Enrique, Rodrigo, Jason
USIL: Cesar
ULima: Percy, Alexandra y Chipana (opcional)



producto (fechas y posible 1-day MVP)

que tan rapido estara listo?
Enrique propone 1-day MVP, si es que falta para la app.


launch (form & entrevistas)

- 60 minutes form -> spam
low cost, high impact, low succes chance

register & waitlist & contacto para entrevista
- entrevistas











reemplaza inmuebles ergonomicos?

Ley / sustento medico



Demo sebastian

SaaS




SaaS added values:

- existen rubros que requieren seguimiento medico detallado



Ancón
Ate
Barranco
Breña
Carabayllo
Chaclacayo
Chorrillos
Cieneguilla
Comas
El Agustino
Independencia
Jesús María
La Molina
La Victoria
Lima (Cercado de Lima)
Lince
Los Olivos
Lurigancho (Chosica)
Lurín
Magdalena del Mar
Miraflores
Pachacámac
Pucusana
Pueblo Libre
Puente Piedra
Punta Hermosa
Punta Negra
Rímac
San Bartolo
San Borja
San Isidro
San Juan de Lurigancho
San Juan de Miraflores
San Luis
San Martín de Porres
San Miguel
Santa Anita
Santa María del Mar
Santa Rosa
Santiago de Surco
Surquillo
Villa El Salvador
Villa María del Triunfo




| Program                               | Focus (Leadership or startup)           | Format & Intensity                                                                           | Eligibility                                                       | Cost                                                | Region                         |
|---------------------------------------|-----------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------|-----------------------------------------------------|--------------------------------|
| **Y Combinator (YC)**                 | Startup         | In-person/Hybrid; 3-month full-time accelerator                                             | Global; highly competitive (~1% acceptance)                       | Free – invests ~$500k for ~7% equity                  | Global (Silicon Valley base)   |
| **Techstars**                         | Startup         | In-person/Hybrid; 12-week intensive accelerator with mentorship                             | Global; no age limit; highly competitive                          | Free – invests ~$120k for ~6% equity                  | Global (50+ programs)          |
| **Thiel Fellowship**                  | Startup         | Remote fellowship; 2-year self-directed program                                             | Global; must be ≤22 years (drop out of school)                      | Free – $100k grant (non-equity)                       | Global (U.S.-based)            |
| **Makers Fellowship**                 | Both            | Hybrid fellowship; intensive training & incubation (multi-month program)                    | Latin American youth (~18–25); highly selective (~1% accepted)       | Free – no cost (funded by partners)                   | Latin America (online + hubs)  |
| **Tony Elumelu Entrepreneurship Program** | Startup   | Online accelerator; 12-week training + mentorship                                            | Africans; age ≥18; startups ≤5 years                               | Free – $5,000 seed grant upon completion            | Africa (54 countries)          |
| **Echoing Green Fellowship**          | Both            | 2-year full-time fellowship supporting social startups                                      | Global; early-stage social entrepreneurs (organization ≤2 years)    | Free – stipend up to ~$90k over 2 years              | Global                         |
| **Young Social Entrepreneurs Global** | Startup         | Hybrid; 6-month part-time program with workshops, mentorship & overseas visit               | Global; ages 18–35                                                 | Free – no fee; grants up to S$20k for top teams       | Global (Singapore-based)       |
| **Halcyon Incubator**                 | Startup         | In-person residency; 5-month full-time incubator for social ventures                          | Global; early-stage social entrepreneurs (<$500k revenue)           | Free – includes residence, mentorship, stipend       | Global (Washington, D.C.)      |
| **MassChallenge**                     | Startup         | Hybrid; 4-month accelerator with mentorship and no equity requirement                         | Global; early-stage startups in any industry                        | Free – zero-equity accelerator (prizes available)    | Global (U.S., Europe, etc.)      |
| **Entrepreneur First (EF)**           | Startup         | In-person cohort; ~6-month program to form co-founding teams and build startups                | Global; early-career tech/entrepreneurial individuals (typically 20s) | Free – talent stipend (e.g., ~\$12k) + investment up to ~\$250k for equity | Global (US, Europe, Asia)       |
| **WEF Global Shapers**                | Leadership      | Ongoing volunteer network; local hubs driving community projects                             | Global; under 30 (typically 18–27)                                  | Free                                               | Global (500+ city hubs)        |
| **One Young World Summit**            | Leadership      | Annual 4-day global summit plus ongoing network                                             | Global; ~18–30 (most delegates under 30)                           | Paid – summit fee & travel (scholarships available)  | Global (rotating host countries)|
| **YALI Regional Leadership Centers**  | Leadership      | In-person, 4-week intensive leadership training                                               | Sub-Saharan Africa; ages 18–35                                      | Free – fully funded (training, travel, lodging)      | Africa (4 regional hubs)       |
| **Young Leaders of the Americas Initiative (YLAI)** | Both   | 5-week fellowship exchange: 4 weeks at a U.S. host company + summit                           | LAC & Canada; ages 25–35 (entrepreneurs/social leaders)              | Free – US Govt. funded (travel, visa, stipend)       | Americas (USA-based program)   |
| **Atlas Corps Fellowship**            | Leadership      | 12–18 months full-time professional fellowship with service and training                      | Global (non-US); young professionals ~23–35                         | Free – stipend provided for living expenses          | Global (placements in US/online)|
| **Acumen Fellows**                    | Leadership      | Part-time, 1-year fellowship with multi-day seminars and group projects                        | Regional cohorts (e.g. East Africa, India); early/mid-career innovators | Free – fully funded                                  | Global (regional programs)     |
| **StartingBloc Institute**            | Leadership      | 5-day intensive leadership bootcamp followed by an alumni network                              | Global; emerging leaders (students/early-career, 20s–30s)             | Paid – tuition ~$1,500 (scholarships available)      | Primarily US (multiple cities) |
| **Venture for America (VFA)**         | Leadership      | 2-year fellowship & job placement; startup training camp plus full-time startup job             | USA only; recent college grads (~21–24, with work authorization)      | Free – fellows earn a salary from their startup employer | United States (various cities) |
| **Global Good Fund Fellowship**       | Both            | 15-month part-time fellowship with leadership coaching and mentoring for social entrepreneurs  | Global; young social enterprise leaders (typically early-career)     | Free – no fee (up to ~$10k project funding provided)  | Global                         |
| **Startmate Accelerator** | Australia & New Zealand (Sydney, Melbourne, NZ; virtual options) | 12 weeks (includes Week 0 offsite setup, weekly meetings, Demo Day, optional events) | Ambitious early-stage, tech-enabled founders from any industry in ANZ | ~$75k investment via SAFE note (mentors invest personal funds) | No program fee (investment provided) | Intensive mentorship; strong community/alumni network; industry‐agnostic; Demo Days; long-term network support; culture of radical honesty and accountability |
| **18startup Fellowship**    | India (headquartered in Coimbatore; bootcamp in Goa)       | 12 weeks intensive program with a 4-day offline bootcamp kickoff and subsequent online sessions | College students and early-stage founders looking to convert ideas into ventures | Fellowship fee grants access to funding opportunities (potential to raise up to ₹25 lakhs) plus technical credits from partners | ₹25,000 (inclusive of taxes)      | Founder-led learning; hands-on mentorship; vibrant community; structured curriculum from ideation to MVP; networking with mentors, investors, and industry experts |
