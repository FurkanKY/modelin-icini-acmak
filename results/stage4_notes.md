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

## 3. tuned lens

logit lens ara katmanı doğrudan çıkış katmanına veriyor. tuned lens (belrose ve ark. 2023) her katman için küçük bir
çevirici öğreniyor. çevirici ara katmanın durumunu son katmanın tahminine en çok benzeyecek biçimde dönüştürüyor.
çeviricileri dört dilde eşit miktarda vikipedi metniyle 1500 adım eğittim (her model için 2000 parça). modelin kendisi değişmiyor.
kod `src/stage4_tunedlens.py`, sonuçlar `data/stage4_tuned_*.json`.

| model | yön | mercek | ingilizce | kaynak | hedef | ingilizce hedefi geçiyor |
|---|---|---|---|---|---|---|
| qwen | fr→tr | logit | 0.075 | 0.013 | 0.035 | 71/93 |
| qwen | fr→tr | tuned | 0.067 | 0.024 | 0.075 | 46/93 |
| qwen | de→tr | logit | 0.098 | 0.006 | 0.028 | 84/105 |
| qwen | de→tr | tuned | 0.075 | 0.038 | 0.071 | 54/105 |
| qwen | tr→fr | logit | 0.166 | 0.003 | 0.126 | 53/93 |
| qwen | tr→fr | tuned | 0.115 | 0.016 | 0.182 | 35/93 |
| kumru | fr→tr | logit | 0.002 | 0.038 | 0.304 | 10/92 |
| kumru | fr→tr | tuned | 0.022 | 0.067 | 0.137 | 21/92 |
| kumru | de→tr | logit | 0.010 | 0.002 | 0.323 | 18/105 |
| kumru | de→tr | tuned | 0.016 | 0.089 | 0.133 | 22/105 |
| kumru | tr→fr | logit | 0.092 | 0.656 | 0.038 | 63/92 |
| kumru | tr→fr | tuned | 0.062 | 0.072 | 0.050 | 54/92 |

tuned lens ile qwen'de ingilizcenin üstünlüğü büyük ölçüde kayboluyor, ingilizce ile hedef dil aşağı yukarı başa baş.
kumru'da da aynısı ters yönde oluyor. türkçeden fransızcaya çevirirken ara katlardaki türkçe logit lens'te %66, tuned lens'te %7.
yani logit lens her modelde o modelin çıkış katmanının ağırlıklı olduğu dili abartıyor. qwen'de bu ingilizce, kumru'da türkçe.

çeviriciler kısa eğitildi, kl değeri adımdan adıma 1.3 ile 2.5 arasında (kumru'da 0.7 ile 1.5) gidip geliyor.
daha uzun eğitim sayıları biraz değiştirebilir.

## ne çıkardım

1. qwen'in ara katları ingilizce kelimelere yakın, kumru'nunkiler türkçeye.
2. qwen'in bu ingilizceyi bir basamak gibi kullandığına dair bir kanıt bulamadım. üç müdahale testinin üçü de olumsuz.
3. en makul açıklama bence şu. ortadaki kavram dilden büyük ölçüde bağımsız ama qwen'in çıkış katmanı ingilizce ağırlıklı,
   o yüzden o kavram ingilizce kelimelere yakın okunuyor. tuned lens bunu destekliyor. aynı etki kumru'da türkçe için var.
   bayazıt ve ark. (2026) da logit lens'in ingilizceyi abarttığını söylüyor.
4. dil yönlerini "o dilin ortalaması eksi bütün dillerin ortalaması" diye bulmak tuzaklı. yönler birbirine karışıyor
   ve silme deneyleri yanıltıcı sonuç verebiliyor.

## açık sorular

- kumru tr→fr'de ingilizceyi silmek zarar verdi (11 doğrudan 5'e). kumru fransızcaya giderken ingilizceyi köprü olarak mı kullanıyor?
- ekli kelimelerde ("evlerimizden") ne oluyor?
