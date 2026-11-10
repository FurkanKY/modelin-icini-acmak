# 4. aşama: model türkçe üretirken içeride ingilizceden geçiyor mu?

10 kasım 2026. modeller qwen3.5-0.8b (çok dilli) ve kumru-2b-base (neredeyse tamamen türkçe eğitilmiş). colab t4.
106 somut isim, dört dilde (ingilizce, türkçe, fransızca, almanca). kod `src/stage4_*.py`, ham sonuçlar `data/stage4_*.json`.

## 1. logit lens (wendler ve ark. 2024'ün çeviri deneyi)

modele birkaç örnekle fransızcadan türkçeye kelime çevirtiyorum ve her katmanda "şu an söyleseydin ne derdin" diye bakıyorum.
görevde hiç ingilizce yok. tablo ara katlardaki (son katmanın %60-90'ı) en yüksek ortalama olasılık.

| model | yön | ingilizce | kaynak | hedef | ingilizce hedefi geçiyor |
|---|---|---|---|---|---|
| qwen | fr→tr | 0.075 | 0.013 | 0.035 | 71/93 |
| qwen | tr→fr | 0.166 | 0.003 | 0.126 | 53/93 |
| qwen | de→tr | 0.098 | 0.006 | 0.028 | 84/105 |
| kumru | fr→tr | 0.002 | 0.038 | 0.304 | 10/92 |
| kumru | tr→fr | 0.092 | 0.656 | 0.038 | 63/92 |
| kumru | de→tr | 0.010 | 0.002 | 0.323 | 18/105 |

qwen'in ara katlarında ingilizce okunuyor, kumru'nunkinde türkçe. sadece türkçesi de qwen'de tek token olan kelimelerde
(su, ev, baş, şehir gibi) sonuç aynı: fr→tr 14/17, de→tr 16/18. yani bu türkçenin harflere bölünmesinden gelmiyor.

## 2. müdahale testleri

logit lens sadece okuma. qwen'in bu ingilizceyi gerçekten kullanıp kullanmadığını üç yolla denedim.

- **dil yönünü silmek.** ingilizce yönünü ara katlardan silince qwen'in türkçe çevirisi çöktü (fr→tr 6 doğrudan 0'a).
  ama qwen'de ingilizce ve türkçe yönleri arasındaki kosinüs -0.60. ingilizce yönünden türkçe payını çıkarıp kalanını
  silince etki kayboldu (6 doğrudan 5'e). ilk sonuç türkçenin dolaylı silinmesinden geliyormuş.
- **yama.** almancadan fransızcaya çeviren bir çalıştırmanın iç durumunu fransızcadan türkçeye çeviren başka bir
  çalıştırmaya yapıştırdım. qwen'de 14-15. katmanlarda kavram türkçe çıkıyor, 16'dan sonra vericinin dili olan fransızca.
  ingilizce hiçbir katmanda %3'ü geçmiyor.
- **kavram enjeksiyonu.** çeviri sırasında başka bir kavramın ingilizce ya da almanca kelime vektörünü ekledim.
  ingilizce vektör modele o kavramın türkçesini söyletmekte almancadan iyi değil, ikisi de etkisiz. sadece 12 kelime,
  kumru'da bu yöntemle test edilebilen kelime çıkmadı.

## ne çıkardım

1. qwen'in ara katları ingilizce kelimelere yakın, kumru'nunkiler türkçeye.
2. qwen'in bu ingilizceyi bir basamak gibi kullandığına dair bir kanıt bulamadım. üç müdahale testinin üçü de olumsuz.
3. en makul açıklama bence şu. ortadaki kavram dilden büyük ölçüde bağımsız ama qwen'in çıkış katmanı ingilizce ağırlıklı,
   o yüzden o kavram ingilizce kelimelere yakın okunuyor. bayazıt ve ark. (2026) da logit lens'in ingilizceyi abarttığını söylüyor.
4. dil yönlerini "o dilin ortalaması eksi bütün dillerin ortalaması" diye bulmak tuzaklı. yönler birbirine karışıyor
   ve silme deneyleri yanıltıcı sonuç verebiliyor.

## açık sorular

- tuned lens ile bakınca qwen'deki ingilizce ne kadar kalıyor?
- kumru tr→fr'de ingilizceyi silmek zarar verdi (11 doğrudan 5'e). kumru fransızcaya giderken ingilizceyi köprü olarak mı kullanıyor?
- ekli kelimelerde ("evlerimizden") ne oluyor?
