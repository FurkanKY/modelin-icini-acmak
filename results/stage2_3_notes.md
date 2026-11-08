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

| deneme | ne yaptım | takılan | doğru | ort. token | takıldı→bitti | bitti→takıldı |
|---|---|---|---|---|---|---|
| düzenlenmemiş | | 34 | 41 | 2013 | | |
| tekrar, α = 1 | 8-22. katmanlar, W − u(uᵀW) | 32 | 39 | 1861 | 17 | 15 |
| tekrar, α = 1.5 | 8-22. katmanlar, mit p10c'nin karşılığı | 41 | 35 | 2098 | 13 | 20 |
| tekrar, 64 sütun | 15. ve 19. katman mlp'si, yönü en çok yazan 64 sütun, α = 1.5 (mit p50c) | 30 | 38 | 1940 | 13 | 9 |

tekrar yönü modeli bozmuyor ama takılmayı da anlamlı biçimde azaltmıyor. mit'nin büyük kazancını bizim ayarlarda göremedim.

bu üç deneme colab oturumu sıfırlandıktan sonra, kütüphaneler yeniden kurularak çalıştı. düzenlenmemiş taban önceki oturumdan. aradaki teknik farklar küçük sapmalara yol açmış olabilir.

bu üç denemenin ham cevapları ben indiremeden colab oturumu bir kez daha sıfırlanınca kayboldu. denemeleri yeniden çalıştırıp `data/` klasörüne ekleyeceğim. sayılar biraz değişirse bu tabloyu da güncelleyeceğim.

## ne çıkardım

1. model takılacaksa bu ilk paragrafta modelin içinden büyük ölçüde belli oluyor.
2. bu sinyal bir "bitirme" ekseni gibi davranıyor. bir yöne itince daha çok, öbür yöne itince daha az takılıyor.
3. takılan cevapların bir kısmında model cevabı biliyor ama bitiremiyor.
4. bitirmeye itmek doğruluğu artırmıyor. model kararsız kaldığı cevabı yanlış bitirebiliyor ve itme iyi giden cevapları da bozuyor.
5. tekrar yönü ayrı bir şey. mit'nin yöntemini onunla tekrarlayınca model bozulmuyor ama düzelmiyor da.

## açık sorular

- mit düşünme modunu açık tuttu ve cevapların %93'ü takılıyordu. bizde %34. döngünün türü farklı olabilir. aynı denemeyi düşünme modu açık yapmak en doğrudan test.
- mit yönü nasıl bulduğunu yayınlamadı. bizim tekrar yönümüz onunkiyle aynı olmayabilir.
- 100 soru, bu boyuttaki etkileri ayırmak için az.
