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

## 4. ekli kelimeler

fransızca "de nos maisons" (ingilizcesi "from our houses") türkçede tek kelime, "evlerimizden". türkçe cevap kökle ("ev")
başlıyor, ingilizce karşılığı edatla ("from"). 20 isim ve 3 hal ("-den", "-de", "-e"), toplam 60 ifade.
ara katlarda türkçe köke, ingilizce edata ve ingilizce isme bakıyorum. kod `src/stage4_suffix.py`, sonuçlar `data/stage4_suffix_*.json`.

- iki model de bu görevde zayıf. qwen 58 ifadenin 2'sini tam doğru yazdı, "evlerimizden" yerine "mehanelerimizden" gibi
  şeyler çıkardı. kumru 8/58. kumru kökü genelde buluyor ama hal ekini karıştırıyor ("ağaçlarımızdan" yerine "ağaçlarımızda").
- ingilizce edat ("from", "in", "to") logit lens'te hiçbir modelde görünmüyor. tuned lens'te qwen'in 16-18. katmanlarında
  %1-2 civarında küçük bir iz var, türkçe kökten biraz yüksek. bu kadar zayıf bir modelde bu büyüklükte bir farkı yorumlamıyorum.
- qwen'de logit lens'te ingilizce isim ("house") 19-22. katmanlarda türkçe kökle aynı seviyede ya da üstünde.
- kumru'da türkçe kök 13. katmandan sonra açıkça yükseliyor, ingilizce iz yok.

sonuç yok sayılır. modeller ekli kelimeyi doğru üretemediği için ara katlarda ne planladıklarını okumak anlamsız kalıyor.
daha güçlü bir modelle (örneğin qwen3.5'in büyük bir sürümü) denemek gerekir.

## 5. kumru ingilizceyi köprü olarak kullanıyor mu?

ilk silme deneyinde kumru tr→fr'de ingilizce yönünü silince doğru cevap 11'den 5'e düşmüştü. ama aynı silme ara
katlardaki ingilizceyi azaltmadı, 0.10'dan 0.32'ye çıkardı. bu qwen'deki tuzağa benziyordu. kumru'da da yönler
birbirine karışık (ingilizce ile fransızca arasında kosinüs -0.45, almanca ile -0.49). üç yoldan baktım.
kod `src/stage4_bridge.py`, sonuçlar `data/stage4_bridge_*.json`.

- **dik yön.** ingilizce yönünden öteki üç dilin yönlerini çıkarıp kalanı sildim. kumru tr→fr'de doğru cevap 11'de
  kaldı. yani ilk düşüşün kaynağı yönün içine karışmış fransızca ve almanca payıymış.
- **kelimeye özel silme.** her kelime için ara katlardan o kelimenin ingilizcesinin çıkış vektörünü sildim.
  kontrol olarak almancasını, yöntemi sınamak için de hedef kelimenin kendisini sildim.

| model | yön | hiçbiri | ingilizce | almanca | hedef kelime | ingilizce (hedeften arındırılmış) | almanca (arındırılmış) |
|---|---|---|---|---|---|---|---|
| kumru | tr→fr | 18/92 | 18 | 16 | 0 | 18 | 16 |
| kumru | fr→tr | 38/92 | 38 | 38 | 1 | 40 | 38 |
| qwen 0.8b | fr→tr | 15/93 | 8 | 15 | 6 | 14 | 17 |
| qwen 0.8b | tr→fr | 45/93 | 29 | 34 | 7 | 40 | 42 |

  hedef kelimeyi silmek cevabı neredeyse sıfırlıyor, yani yöntem bir şeyi bozabiliyor. kumru'da ingilizceyi silmek
  hiçbir şey değiştirmiyor. qwen 0.8b'de ingilizceyi silmek ilk bakışta zarar veriyor (45'ten 29'a) ama ingilizce
  kelime vektörü hedef kelimeyle biraz örtüşüyor (özellikle fransızcada ortak kökler yüzünden). o örtüşen kısmı
  çıkarınca etki almancanınkiyle aynı seviyeye iniyor (40'a karşı 42).
- **okuma.** kumru tr→fr'de ingilizce ara katlarda gerçekten görünüyor, tuned lens'te bile (0.062, almanca 0.000).
  92 kelimenin 86'sında ingilizce almancanın önünde.

sonuç olarak kumru fransızcaya giderken ara katlarda ingilizce kelimeye yakın bir yerden geçiyor ama bu ingilizceyi
silmek çeviriyi bozmuyor. köprü gibi görünüyor, köprü gibi çalışmıyor.

## 6. daha büyük model: qwen3.5-4b

0.8b ekli kelimelerde çok zayıftı, o yüzden aynı deneyleri qwen3.5-4b ile tekrarladım (float16, t4'e sığıyor).
4b için de ayrı bir tuned lens eğittim (800 adım, dört dilde 1600 vikipedi parçası).

**çeviri.** 4b fr→tr'de 93 kelimenin 78'ini doğru başlatıyor (0.8b 15/93).

| yön | mercek | ingilizce | kaynak | hedef | ingilizce hedefi geçiyor |
|---|---|---|---|---|---|
| fr→tr | logit | 0.386 | 0.005 | 0.022 | 92/93 |
| fr→tr | tuned | 0.179 | 0.009 | 0.092 | 64/93 |
| de→tr | logit | 0.375 | 0.008 | 0.028 | 103/105 |
| de→tr | tuned | 0.143 | 0.016 | 0.089 | 68/105 |
| tr→fr | logit | 0.492 | 0.001 | 0.155 | 83/93 |
| tr→fr | tuned | 0.188 | 0.004 | 0.274 | 52/93 |

logit lens'te 4b'nin 26-29. katmanları neredeyse tamamen ingilizce (%39), türkçe 30. katmandan sonra geliyor.
tuned lens bunu yarıya indiriyor ama 0.8b'deki gibi silmiyor. türkçeye çevirirken ingilizce hâlâ hedefin iki katı.
yani büyük modelde ara katlardaki ingilizce mercekten gelen bir yanılsamadan daha fazlası.

**ekli kelimeler.** 4b 58 ifadenin 40'ını tam doğru yazdı ("de nos maisons" → "evlerimizden"). bu sefer ara katlara
bakmak anlamlı.

| katman | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
|---|---|---|---|---|---|---|---|
| ingilizce isim ("house"), logit | 0.254 | 0.334 | 0.372 | 0.195 | 0.039 | 0.004 | 0.000 |
| ingilizce isim, tuned | 0.041 | 0.087 | 0.150 | 0.080 | 0.011 | 0.001 | 0.000 |
| türkçe kök ("ev"), tuned | 0.010 | 0.012 | 0.022 | 0.056 | 0.235 | 0.439 | 0.715 |
| ingilizce edat ("from"), tuned | 0.001 | 0.001 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 |

ara katlarda ingilizce isim türkçe kökten önce ve daha güçlü görünüyor. ingilizce edat ise hiçbir katmanda yok.
ingilizce cümle "from our houses" diye edatla başlıyor. model ingilizce cümle kurup çeviriyor olsaydı "from"un bir
yerde görünmesini beklerdim. görünen tek şey içerik kelimesi. bu da ingilizcenin bir cümle planı olarak değil,
kavramın adı olarak durduğunu düşündürüyor.

**silme.** 5. bölümdeki kelimeye özel silmeyi 4b'de tekrarladım. ara katlardan (16-27) o kavramın ingilizce
kelimesinin çıkış vektörünü siliyorum. "arındırılmış" sürümlerde vektörden önce hedef türkçe kelimenin payını
çıkarıyorum. "başka ingilizce" ilgisiz bir kavramın ingilizcesi, herhangi bir ingilizce kelimeyi silmenin zarar
verip vermediğini görmek için.

| yön | hiçbiri | ingilizce | ingilizce (arındırılmış) | almanca (arındırılmış) | başka ingilizce | türkçe hedef |
|---|---|---|---|---|---|---|
| fr→tr | 77/93 (0.71) | 64 (0.48) | 67 (0.53) | 76 (0.69) | 76 (0.72) | 65 (0.60) |
| de→tr | 91/105 (0.76) | 71 (0.53) | 76 (0.58) | 91 (0.74) | 91 (0.76) | 79 (0.64) |

parantez içindeki sayı son katmanda doğru cevaba verilen ortalama olasılık.
ingilizce kelimeyi silmek 4b'nin türkçe çevirisini belirgin şekilde bozuyor, hedef türkçe kelimeyi silmekten bile
fazla. almancayı ya da ilgisiz bir ingilizce kelimeyi silmek hiçbir şey yapmıyor. hedef kelimeyle örtüşen payı
çıkarınca da etki büyük ölçüde kalıyor (77'den 67'ye). türkçenin kendisini silmek daha az etkili çünkü bu katmanlarda
türkçe henüz ortaya çıkmamış, 30. katmandan sonra geliyor.

bu, bütün aşamada ilk olumlu müdahale sonucu. 0.8b'de ve kumru'da aynı test hiçbir şey göstermedi.

bir şüphe kaldı. ingilizce kelime çoğu zaman kaynak kelimeye benziyor (almanca "haus" ile "house"). ingilizce vektörü
silerken kaynak kelimenin bilgisini de silmiş olabilirim. bunu ayırmak için kaynak kelimeyi silmeyi ve ingilizce
vektörden kaynağın payını da çıkarmayı deneyeceğim. o test bitene kadar bu sonucu kesin saymıyorum.

## ne çıkardım

1. qwen'in ara katları ingilizce kelimelere yakın, kumru'nunkiler türkçeye.
2. küçük qwen'de (0.8b) bu ingilizce büyük ölçüde mercekten geliyor. tuned lens ile ingilizce ve türkçe başa baş,
   müdahale testlerinin hiçbiri ingilizcenin kullanıldığını göstermedi. qwen'in çıkış katmanı ingilizce ağırlıklı,
   o yüzden ortadaki kavram ingilizce kelimelere yakın okunuyor. aynı etki kumru'da türkçe için var.
3. büyük qwen'de (4b) tablo değişiyor. ingilizce tuned lens'te de hedefin iki katı ve o kavramın ingilizce kelimesini
   ara katlardan silmek türkçe çeviriyi bozuyor. almancayı ya da ilgisiz bir ingilizce kelimeyi silmek bozmuyor.
   yani 4b türkçeye çevirirken ara katlarda kavramı ingilizce kelimeye yakın bir biçimde tutuyor ve büyük ihtimalle
   bunu kullanıyor. kaynak kelime kontrolü henüz yapılmadı (bkz. 6. bölüm).
4. bu bir ingilizce cümle planı gibi görünmüyor. "evlerimizden" üretirken ara katlarda "house" var, "from" yok.
   ingilizce kavramın adı olarak duruyor, kelime sırası türkçenin.
5. kumru fransızcaya giderken ara katlarda ingilizce görünüyor ama silmek bir şey değiştirmiyor. köprü değil.
6. dil yönlerini "o dilin ortalaması eksi bütün dillerin ortalaması" diye bulmak tuzaklı. yönler birbirine karışıyor
   ve silme deneyleri yanıltıcı sonuç verebiliyor. kelimeye özel silme, olumlu kontrol (hedef kelimeyi silmek) ve
   ilgisiz kelime kontrolüyle çok daha temiz sonuç veriyor.

## açık sorular

- 0.8b ile 4b arasındaki fark modelin büyüklüğünden mi geliyor? arada qwen3.5-2b var, aynı testlerden geçirmek lazım.
- 4b tuned lens 800 adım eğitildi, kl hâlâ 2.4 ile 3.2 arasında gidip geliyor. daha uzun eğitim ingilizceyi biraz daha
  küçültebilir. silme sonucu mercekten bağımsız olduğu için ana bulguyu değiştirmez.
- silme sadece kelimenin ilk token'ının çıkış vektörünü kaldırıyor. kavramın ingilizce temsili bundan daha geniş olabilir,
  yani gerçek etki ölçtüğümden büyük olabilir.
