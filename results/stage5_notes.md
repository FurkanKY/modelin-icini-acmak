# aşama 5: türkçe matematik soruları — sonuçlar

tahminler deneyden önce `results/stage5_tahminler.md`'ye yazıldı, o dosyaya dokunulmadı. kurulum orada.
ham cevaplar `data/stage5_*.json`, analiz `src/stage5_analiz.py` → `data/stage5_analiz.json`.
tek oturum, colab t4, 11 kasım 2026. en_off 3. aşamadaki tabanla birebir aynı çıktı (41 doğru, 33 takılan, 1945 token).

## 1. doğruluk

| koşul | soru | talimat | düşünme | doğru | sınıra dayanan |
|---|---|---|---|---|---|
| en_off | en | en | kapalı | 41 | 33 |
| tr_off | tr | tr | kapalı | 9 | 46 |
| trq_eninst_off | tr | en | kapalı | 8 | 58 |
| geri_off | tr→en geri çeviri | en | kapalı | 30 | 34 |
| en_on (stage3_think_base_v2) | en | en | açık | 10 | 100 |
| tr_on | tr | tr | açık | 13 | 96 |
| zorla_en | en | en | açık, döngüde kes | 34 | 62 (1024 token'da bitmeyen) |
| zorla_tr | tr | tr | açık, döngüde kes | 18 | 74 (1024 token'da bitmeyen) |

## 2. eşli karşılaştırmalar (b − a, soru başına, kesin mcnemar, %95 bootstrap aralık)

| a | b | a doğru b yanlış | b doğru a yanlış | fark [%95] | p |
|---|---|---|---|---|---|
| en_off | tr_off | 35 | 3 | −32 [−42, −22] | 6.7e-08 |
| en_off | geri_off | 16 | 5 | −11 [−20, −3] | 0.027 |
| tr_off | geri_off | 3 | 24 | +21 [+12, +30] | 4.9e-05 |
| tr_off | trq_eninst_off | 3 | 2 | −1 [−5, +3] | 1 |
| en_on | tr_on | 3 | 6 | +3 [−3, +9] | 0.51 |
| tr_off | tr_on | 5 | 9 | +4 [−3, +11] | 0.42 |
| tr_on | zorla_tr | 1 | 6 | +5 [0, +10] | 0.12 |
| zorla_en | zorla_tr | 19 | 3 | −16 [−25, −7] | 0.00086 |
| en_off | zorla_tr | 27 | 4 | −23 [−33, −13] | 3.4e-05 |

32 puanlık kaybın 11'i (aralık 3-20) çeviri zincirinde (bezir çevirisi + geri çeviri) kayboluyor, 21'i (12-30) aynı içeriği türkçe sormaktan geliyor. geri çeviri iki çeviri adımı içerdiği için çeviri payını büyük gösteriyor olabilir, yani 21 dil payı için alt sınıra yakın.

## 3. puanlama hatası değil

- tr_off'ta en doğru / tr yanlış 35 soruya baktım: hepsi gerçek yanlış cevap (ör. `12` yerine `\frac{12(x-2)}{x^2+4}`, `(3/2, −13)` yerine `(3/2, 3)`). biçim yüzünden kaçan yok.
- bitmiş cevaplarda doğruluk: en_off 40/67 (%60), geri_off 28/66 (%42), tr_off 9/54 (%17), trq_eninst_off 7/42 (%17). yani düşüş sadece daha çok takılmaktan değil, biten cevaplar da yanlış.
- `\boxed{}` olan cevap: tr_off 52, en_off 68.
- zorluk seviyesine göre (en_off / tr_off doğru): 1: 4/2 (6 soru), 2: 8/2 (12), 3: 13/4 (23), 4: 13/1 (37), 5: 3/0 (22). kayıp her seviyede, en çok 3-4'te.

## 4. dil

100 token'lık parçalar, latex atılarak, türkçe/ingilizce sözcük ve türkçe karakter sayılarak sınıflandı (`parca_dili`). çok formüllü parçalar "belirsiz".

- düşünme kapalı: tr_off cevaplarının 99'u türkçe. **ingilizce talimat (trq_eninst_off) bunu değiştirmiyor, yine 99 türkçe.** geri_off ve en_off 100 ingilizce.
- düşünme açık, türkçe soru (tr_on, döngü öncesi kısım): 808 ingilizce, 31 türkçe, 516 belirsiz parça. sınıflanabilen parçaların %96'sı ingilizce. soru başına çoğunluk: 75 ingilizce, 25 belirsiz, 0 türkçe. ingilizce soruda (en_on) 855 / 1 / 638.
- tr_on'da sadece 4 cevap `</think>` ile düşünmeyi kapatıyor.
- zorla_tr'de `</think>`'ten sonra yazılan son cevap 99/100 türkçe. zorla_en 100 ingilizce.

## 5. döngü (token bazlı: 8 token'lık parça, 256 token pencere, %90 tekrar)

| | döngüye giren | medyan başlangıç (token) | sınıra dayanan |
|---|---|---|---|
| en_on | 96 | 1184 | 100 |
| tr_on | 94 | 1108 | 96 |

döngü türkçeye özgü değil, ingilizcedekiyle aynı oranda ve aynı yerde.

## 6. kayıp nerede oluyor (zorlama ayrıştırması)

kesilen düşüncenin içindeki son cevabı (math-verify'ın düşünce metninden çıkardığı) puanladım:

| | düşünce kesildiği anda doğru | zorlama sonrası doğru | düşünce doğru → zorla yanlış | düşünce yanlış → zorla doğru |
|---|---|---|---|---|
| en | 14 | 34 | 2 | 22 |
| tr | 12 | 18 | 2 | 8 |

ortalama kesim noktası 1449 ve 1421 token, aynı. yani kesildiği ana kadar ingilizce düşünce iki soru dilinde aynı kalitede. fark son cevap yazılırken çıkıyor: ingilizce cevap 22 soruyu kurtarıyor, türkçe cevap 8. türkçe cevaplar 1024 token'da daha az bitiyor (26'ya karşı 38) ve örneklerde model türkçe cümlelerde yeniden döngüye giriyor ("bir doğruya göre bir doğruya göre ...", "x tam sayı olmalı ... x tam sayı olmalı").

## 7. ayrı raporlanan sorular

- sayısı türkçe biçimli (242, 263, 273): 242 ve 263 hiçbir koşulda doğru değil. 273 sadece en_off'ta doğru. sonuçları değiştirmiyor.
- metin cevaplı: 227 (harf) en_off, geri_off, zorla_en doğru. 255 (harf) geri_off, zorla_tr doğru. 257 ("inches") en_off, geri_off, tr_on, zorla_tr, zorla_en doğru. 296 (harf) sadece en_off. tr_off ve trq_eninst_off dördünde de yanlış.

## 8. token şişmesi

soru metni ingilizce ortalama 78.2, türkçe 89.4 token (1.14 kat). cevap uzunluğu: en_off 1945, tr_off 2208, trq_eninst_off 2649, tr_on 3997 token. şişme küçük, 32 puanlık farkı açıklamaz.

## 9. tahminlerle karşılaştırma

| tahmin | sonuç | tuttu mu |
|---|---|---|
| tr_off 28-35 (fark 6-13) | 9 (fark 32) | hayır, kayıp tahminin 2.5-5 katı |
| geri_off en_off'un ±5'i, kaybın çoğu dilden | 30 (−11). kaybın 21/32'si dilden | kısmen: çoğu dilden ama çeviri de 11 puan götürüyor |
| tr_off sınıra dayanan 35-50 | 46 | evet |
| tr_on ≥90 sınır, ≥80 döngü, doğru ~10 ya da altı | 96, 94, 13 | ilk ikisi evet, doğru biraz üstünde |
| düşüncenin >%70'i ingilizce, son cevap da çoğunlukla ingilizce | düşüncede sınıflanan parçaların %96'sı ingilizce. son cevap 99/100 türkçe | ilk yarı evet, ikinci yarı hayır |
| tr_zorla < en_zorla ama > 2 × tr_on | 18 < 34 evet, 18 < 26 | yarı yarıya |
| ingilizce talimat tr_off'tan biraz iyi | 8 vs 9, cevaplar yine türkçe | hayır |

## 10. sınırlar

- tek model (0.8b), 100 soru, greedy. 100 soruda ±10 puan civarı fark ancak ayırt ediliyor.
- türkçe çeviri bezir/MATH-500-multilingual'dan, 5 soruda düzeltildi (`data/stage5_tr_duzeltmeler.json`). çeviri kalitesi geri_off ile kısmen ölçüldü.
- geri çeviriyi sadece türkçe metni gören başka bir model (claude sonnet) yaptı. ingilizce aslı görmedi.
- dil sınıflandırıcı basit (sözcük listesi + türkçe karakter). formül ağırlıklı parçalar belirsiz kalıyor, oranlar o yüzden sınıflanabilen parçalar üzerinden de verildi.
- zorla_en burada token bazlı döngü başında kesiyor. 3. yazıdaki 35 kelime bazlı kesimdi. 34 ile 35 aynı sonuç.
