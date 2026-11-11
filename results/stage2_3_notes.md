# 2. aşama (200 soru) ve 3. aşama: notlar

8 kasım 2026. ayarlar 1. aşamayla aynı: qwen3.5-0.8b, greedy, düşünme modu kapalı, 4096 token, colab t4. puanlama math-verify ile.

## 2. aşama: 200 soru

math-500'ün ilk 200 sorusu. 91'i doğru (%45.5), 72'si 4096 token'da bitmedi (%36).

| zorluk | soru | doğru | takılan |
|---|---|---|---|
| 1 | 21 | 15 | 3 |
| 2 | 44 | 30 | 12 |
| 3 | 41 | 19 | 12 |
| 4 | 44 | 19 | 16 |
| 5 | 50 | 8 | 29 |

72 takılmanın 16'sı birebir tekrar, kalanında model aynı şüphenin etrafında farklı cümlelerle dönüyor. biten cevapların yarısı 624 token'ın altında.

ilk 50 soru bu sefer 18 takılan ve 23 doğru çıktı, 1. aşamada 19 ve 20'ydi. arada colab'daki transformers sürümü değişti, greedy'de bile bu kadar sapma olabiliyor.

### takılmayı önceden görmek

her cevabın ilk 512 token'ında, her katmanda, takılanların ortalamasından bitenlerin ortalamasını çıkardım ve bu yönle leave-one-out tahmin yaptım.

- hep "takılmaz" demek %64, sadece zorluğa bakmak %68.
- 5. ile 20. katman arası %76-80. en iyisi 15. katman, %79.5. mit'nin en iyi düzenlemesi de 15. ve 19. katmandaydı.
- pilotta seçtiğim 11. katmanı sadece yeni 150 soruda (50-199) denedim: %75, şans %64.
- aynı 200 soruda yönün bilip zorluğun bilemediği 44 soru var, tersi 21 (iki yönlü binom testi p ≈ 0.006).
- aynı zorluk seviyesi içinde: seviye 2'de %89 (şans %73), 3'te %83 (%71), 4'te %64 (%64), 5'te %84 (%58). 4. seviyede işe yaramıyor, sebebini bilmiyorum.
- yön skoru ile zorluk arasındaki korelasyon 0.54.

200 cevabı 15. katmandaki skora göre sıralayıp 40'arlık 5 gruba bölünce gerçekten takılanlar: 1, 4, 9, 22, 36.

"takılıp doğru" görünen 3 cevap var. bunlarda model doğru sayıyı metne yazmış ama cevabı bitirmemiş, math-verify metnin içinden yakalıyor. 5. soru (altıgen) bunlardan biri: 42'yi buluyor, "final check" diyor ve 3000 kelime boyunca kontrol etmeyi bırakamıyor.

## 3. aşama: yönle oynamak

yön 0-199. sorulardan bulundu, deneme modelin hiç görmediği 200-299. sorularda. her satır 100 soru.

### bitirme yönü (2. aşamadaki yön, 15. katman)

| deneme | ne yaptım | takılan | doğru | ort. token | takıldı→bitti | bitti→takıldı |
|---|---|---|---|---|---|---|
| düzenlenmemiş | | 34 | 41 | 2013 | | |
| sil | 8-22. katmanların dikkat ve mlp çıkışlarında W − u(uᵀW) | 72 | 28 | 3211 | 5 | 43 |
| ters çevir | aynısı 1.5 katsayıyla (mit'nin formülü) | 93 | 25 | 3844 | 0 | 59 |
| hafif it | 15. katman çıkışına her token'da −0.3·u | 25 | 39 | 1686 | 16 | 7 |
| güçlü it | aynısı −0.6·u | 21 | 30 | 1577 | 19 | 6 |

- silmek ve ters çevirmek takılmayı artırıyor, "biten" tarafa itmek azaltıyor. yön modelin bitirme kararında gerçekten kullanılıyor.
- ters çevirince 18 cevapta model doğru sayıyı yazıyor ama bitiremiyor.
- hafif itmede takılmaktan kurtulan 16 cevabın 8'i doğru bitiyor. ama başka 15 doğru cevap yanlışa dönüyor, 13 yanlış doğruya. net doğruluk değişmiyor.
- güçlü itmede kurtulan 19 cevabın 5'i doğru, doğruluk 41'den 30'a düşüyor.
- işaret testleri: hafif itmede 16'ya 7 (p ≈ 0.09), güçlü itmede 19'a 6 (p ≈ 0.015).

### tekrar yönü (mit'nin yönüne yakın olması beklenen)

200 cevabın 512. token'dan sonraki her token'ını iki gruba ayırdım. onunla biten 8 token'lık parça cevapta daha önce geçmişse "tekrar" (126.111 token), geçmemişse "yeni" (185.213 token). yön iki grubun ortalama farkı.

- ayrışma en net 11. katmanda, orta katlar yine en güçlü bölge.
- 15. katmanda bu yön ile bitirme yönü arasındaki kosinüs benzerliği 0.14, neredeyse dik. iki ayrı şey.

yeniden çalıştırma (10 kasım, `src/stage3_think.py`, hepsi aynı oturumda, taban dahil). ham cevaplar `data/stage3_base_v2.json`, `data/stage3_rep_*.json`.

| deneme | ne yaptım | takılan | doğru | ort. token | takıldı→bitti | bitti→takıldı |
|---|---|---|---|---|---|---|
| düzenlenmemiş (v2) | | 33 | 41 | 1945 | | |
| tekrar, α = 1 | 8-22. katmanlar, W − u(uᵀW) | 33 | 38 | 1966 | 13 | 13 |
| tekrar, α = 1.5 | 8-22. katmanlar, mit p10c'nin karşılığı | 35 | 36 | 1912 | 12 | 14 |
| tekrar, 64 sütun | 15. ve 19. katman mlp'si, en çok yazan 64 sütun, α = 1.5 (mit p50c) | 30 | 39 | 1948 | 8 | 5 |

tekrar yönü birebir aynı çıktı (126.111 / 185.213 token, bitirme yönüyle kosinüs 0.143).

ilk çalıştırma (8 kasım, ham cevapları kayıp):

| deneme | ne yaptım | takılan | doğru | ort. token | takıldı→bitti | bitti→takıldı |
|---|---|---|---|---|---|---|
| düzenlenmemiş | | 34 | 41 | 2013 | | |
| tekrar, α = 1 | 8-22. katmanlar, W − u(uᵀW) | 32 | 39 | 1861 | 17 | 15 |
| tekrar, α = 1.5 | 8-22. katmanlar, mit p10c'nin karşılığı | 41 | 35 | 2098 | 13 | 20 |
| tekrar, 64 sütun | 15. ve 19. katman mlp'si, yönü en çok yazan 64 sütun, α = 1.5 (mit p50c) | 30 | 38 | 1940 | 13 | 9 |

tekrar yönü modeli bozmuyor ama takılmayı da anlamlı biçimde azaltmıyor. mit'nin büyük kazancını bizim ayarlarda göremedim.

bu üç deneme colab oturumu sıfırlandıktan sonra, kütüphaneler yeniden kurularak çalıştı. düzenlenmemiş taban önceki oturumdan. aradaki teknik farklar küçük sapmalara yol açmış olabilir.

bu üç denemenin ham cevapları ben indiremeden colab oturumu bir kez daha sıfırlanınca kayboldu. yukarıdaki yeniden çalıştırma bunların yerini alıyor. sayılar birkaç soru oynadı, en büyük fark α = 1.5'te (takılan 41 yerine 35). sonuç aynı.

### düşünme modu açık

mit düşünme modunu açık tutmuştu. aynı 100 soruyu (200-299) `enable_thinking=True` ile sordum, gerisi aynı (greedy, 4096 token).
"birebir döngü": 200 kelimelik bir pencerenin en az %90'ı daha önce geçmiş 8 kelimelik parçalardan oluşuyor (`src/stage3_think_analiz.py`).
ham cevaplar `data/stage3_think_*.json`.

| deneme | doğru | sınıra dayanan | birebir döngü | döngü başlangıcı (medyan kelime) |
|---|---|---|---|---|
| düşünme kapalı (base_v2) | 41 | 33 | 12 | 718 |
| düşünme açık | 10 | 100 | 95 | 524 |
| düşünme açık, bitirme yönüne −0.3·u | 11 | 99 | 91 | 574 |
| düşünme açık, tekrar yönü α = 1.5 | 11 | 99 | 95 | 361 |

- düşünme modunda 100 cevabın 100'ü sınıra dayanıyor, 95'i birebir döngüde. mit'nin %93'ü büyük ihtimalle bu.
- 10 doğru cevabın hepsi bitmeyen cevaplarda. model cevabı buluyor, sonra aynı kontrol satırlarını tekrar ediyor (ilk soruda 40'ı buluyor).
- düşünme bloğu 4096 token'da hiçbir cevapta kapanmıyor. mit 32.768 token kullandı. birebir döngüye giren bir cevabın greedy'de daha uzun sınırla kurtulması pek olası değil ama bunu test etmedim.
- iki yön de düşünme modundaki döngüyü kırmıyor. tekrar yönünü silmek döngüyü daha erken başlatıyor. yönler düşünme kapalı cevaplardan bulundu, düşünme bloğundaki durumu yakalamıyor olabilirler.

## ne çıkardım

1. model takılacaksa bu ilk paragrafta modelin içinden büyük ölçüde belli oluyor.
2. bu sinyal bir "bitirme" ekseni gibi davranıyor. bir yöne itince daha çok, öbür yöne itince daha az takılıyor.
3. takılan cevapların bir kısmında model cevabı biliyor ama bitiremiyor.
4. bitirmeye itmek doğruluğu artırmıyor. model kararsız kaldığı cevabı yanlış bitirebiliyor ve itme iyi giden cevapları da bozuyor.
5. tekrar yönü ayrı bir şey. mit'nin yöntemini onunla tekrarlayınca model bozulmuyor ama düzelmiyor da.
6. mit ile benim aramdaki fark düşünme modundan geliyor. düşünme açıkken model neredeyse her soruda birebir döngüye giriyor. mit'nin kazancı büyük ihtimalle bu döngüyü kırmaktan geliyor ve benim iki yönüm bunu yapamıyor.

## açık sorular

- tekrar yönünü düşünme modu açık cevaplardan çıkarmak. döngü düşünme bloğunun içinde oluyor, yön de oradan bulunmalı.
- 32.768 token sınırıyla düşünme modunda kaç cevabın gerçekten bittiği (t4'te 100 soru için çok uzun).
- mit yönü nasıl bulduğunu yayınlamadı. bizim tekrar yönümüz onunkiyle aynı olmayabilir.
- 100 soru, bu boyuttaki etkileri ayırmak için az.

## kontroller (11 kasım, `src/stage3_kontrol.py`, `src/istatistik.py`)

eleştiriye cevap veren denemeler. özetler `data/stage3_kontrol_summary.json`. rand2_guclu, think_reppen ve think_16k'nin ham cevapları colab oturumu sıfırlanınca kayboldu, sayıları özette duruyor.

| deneme | ne | takılan | doğru | not |
|---|---|---|---|---|
| rand1_ters / rand2_ters | rastgele yön, 8-22. kat, α = 1.5 | 32 / 40 | 36 / 35 | bitirme yönünde 93 |
| rand1_guclu / rand2_guclu | rastgele yön, 15. kat, −0.6·u | 41 / 38 | 37 / 34 | bitirme yönünde 21 |
| ext_base / ext_steer03 | 300-399, taban / hafif it | 29 / 27 | 47 / 42 | geçiş 12/10, p 0.83 |
| think_reppen | düşünme açık, repetition_penalty 1.1 | 97 | 16 | birebir döngü 62 (95'ten), medyan 1083. kelime, 3 think kapandı |
| think_16k | düşünme açık, 200-224, 16.384 token | 25/25 | 3 | aynı 25'te 4096: 3 doğru. think hiç kapanmadı, döngü aynı kelimede |
| think_train | düşünme açık, 0-99 (yön için) | 100 | 15 | |
| think_base_v2 | düşünme açık, aynı oturumda taban | 100 | 10 | ilk çalıştırmayla birebir aynı (95 döngü, 524) |
| think_newrep_a10 / a15 | düşünme modundan tekrar yönü, 8-22. kat | 100 / 100 | 14 / 6 | döngü 98 / 99, medyan 451 / 398 |

- etki yöne özgü. rastgele yönler takılmayı 32-41'de bırakıyor. rastgele yönler doğru sayısını 4-7 düşürüyor, bu genel bir bedel.
- hafif itme 200-299'da 16/7 (p 0.09), 300-399'da 12/10 (p 0.83), birlikte 28/17 (p 0.14). zayıf. güçlü itme 19/6 (p 0.015).
- düşünme modu tekrar yönü: 297.038 tekrar / 61.362 yeni an, bitirme yönüyle kosinüs 0.091. silmek döngüyü kırmıyor, erkene çekiyor. eski tekrar yönünde de aynısı olmuştu.
- tekrar cezası döngüyü kısmen kırıyor ama model aynı şüpheyi başka kelimelerle sürdürüyor.
- daha uzun sınır hiçbir şey değiştirmiyor. "4096 yetmiyor" itirazı geçersiz.

### döngüyü zorla kesmek ve örnekleme (11 kasım, `src/stage3_zorla.py`, `think_sample`)

| deneme (düşünme açık, 200-299) | sınır | birebir döngü | medyan | doğru |
|---|---|---|---|---|
| greedy (think_base_v2) | 100 | 95 | 524 | 10 |
| örnekleme t 0.6, top_p 0.95, top_k 20, tohum 0 | 100 | 61 | 1123 | 11 |
| döngü başında kes + `</think>`, 1024 token cevap | | | | **35** |
| 4096 sonunda kes + `</think>` | | | | 33 |

- zorla kesme: 25 yanlış→doğru, 0 doğru→yanlış. 1024 token'da 40 cevap bitti, 35 alt sınır. sadece `</think>` sonrası puanlandı.
- döngüde kesmek ile sonda kesmek arasında fark yok (30 ikisinde, 5 sadece döngüde, 3 sadece sonda).
- düşünme kapalı 41 > zorla kesilmiş düşünme 35. bu boyutta düşünme modu kazanç getirmiyor.
- örnekleme birebir kopyayı azaltıyor ama bitirmiyor (1/100 think kapandı). sorun greedy'nin yan ürünü değil.
- ham dosyalar: `data/stage3_zorla_{dongu,son}.json`, `data/stage3_zorla_summary.json`, `data/stage3_think_sample.json`.
