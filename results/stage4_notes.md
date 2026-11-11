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

bu, bütün aşamada ilk olumlu müdahale sonucu. kumru'da aynı test hiçbir şey göstermedi, 0.8b'de ise etki ölçülemeyecek kadar küçüktü.

**kaynak kelime kontrolü.** bir şüphe vardı. ingilizce kelime çoğu zaman kaynak kelimeye benziyor (almanca "haus" ile
"house"). ingilizce vektörü silerken kaynak kelimenin bilgisini de silmiş olabilirdim. bunu ayrı bir çalıştırmayla
test ettim (kod `src/stage4_source.py`, sonuç `data/stage4_source_Qwen3.5-4B.json`). bu sefer her kelimenin sonucunu
ayrı kaydettim, böylece farkın şans eseri olup olmadığına da bakabildim. kontrol dili fr→tr'de almanca, de→tr'de
fransızca. ikinci yönde fransızcanın da eklenmesiyle bazı kelimeler elendi, o yüzden de→tr'de 93 kelime var ve
sayılar yukarıdaki tabloyla doğrudan karşılaştırılamaz.

| silinen | fr→tr doğru | p düşüşü [%95 aralık] | de→tr doğru | p düşüşü [%95 aralık] |
|---|---|---|---|---|
| hiçbiri | 77/93 | | 80/93 | |
| türkçe hedef | 65 | 0.110 [0.081, 0.141] | 72 | 0.116 [0.086, 0.148] |
| kaynak kelime | 75 | 0.030 [0.014, 0.049] | 80 | 0.037 [0.019, 0.061] |
| ingilizce | 64 | 0.230 [0.185, 0.277] | 65 | 0.217 [0.172, 0.265] |
| ingilizce, hedef ve kaynaktan arındırılmış | 70 | 0.117 [0.082, 0.153] | 73 | 0.092 [0.056, 0.129] |
| kontrol dili, arındırılmış | 76 | 0.017 [0.007, 0.028] | 80 | 0.000 [-0.008, 0.008] |
| ilgisiz ingilizce, arındırılmış | 76 | -0.001 [-0.003, 0.001] | 81 | -0.001 [-0.004, 0.002] |

"p düşüşü" doğru cevaba verilen olasılığın ne kadar azaldığı. aralık, kelimeleri yeniden örnekleyerek (bootstrap)
bulunan %95 güven aralığı.

- kaynak kelimeyi silmek çok az zarar veriyor (0.03). model kaynak kelimeyi bu yönde pek tutmuyor.
- ingilizce vektörün %93-95'i hedefin ve kaynağın dışında kalıyor. o pay silindiğinde de olasılık 0.09-0.12 düşüyor.
  kontrollerde düşüş 0.02'nin altında. aralıklar birbirinden tamamen ayrı.
- kelimeleri kaynak ile ingilizcesinin yazılışı benziyor mu diye ikiye ayırdım (`src/stage4_source_analiz.py`).
  fr→tr'de benzemeyen 83 kelimede (livre/book, chien/dog) arındırılmış ingilizceyi silmek olasılığı 0.112 düşürüyor.
  de→tr'de benzemeyen 57 kelimede (buch/book, hund/dog) 0.120. benzer kelimelerde (haus/house) etki daha küçük
  (de→tr 0.047). yani etki ingilizcenin kaynağa benzemesinden gelmiyor. benzemeyen kelimelerde de tam gücüyle var.
- doğru cevap sayısında fark daha küçük (77'den 70'e, 80'den 73'e) ve mcnemar testinde sınırda (p 0.04 ile 0.15
  arası). olasılıktaki düşüş net, en üstteki cevabın değişmesi ise birkaç kelimede oluyor.

sonuç olarak kaynak şüphesi düştü. 4b'de o kavramın ingilizce kelimesinin çıkış yönü, türkçe çeviri için gerekli.
aynı kavramın almancası ya da ilgisiz bir ingilizce kelime bu işi görmüyor.

bunun tam olarak ne demek olduğunu da sınırlamak lazım. bu test ara katlardaki kavramın ingilizce kelimeye hizalı
bir yönde durduğunu ve modelin o yönü kullandığını gösteriyor. "model kafasında ingilizce cümle kuruyor" demiyor
(ekli kelimelerde "from" hiç görünmedi). "kavram dilden bağımsız ama ingilizceye en yakın görünüyor" ile "kavram
ingilizce" arasındaki farkı da tam ayıramıyor. aynı kavramın almancasının işe yaramaması ingilizceye özgü bir şey
olduğunu düşündürüyor.

## 7. aradaki boyut: qwen3.5-2b ve üç boyut yan yana

0.8b ile 4b arasındaki fark büyüklükten mi geliyor diye qwen3.5-2b'yi aynı testlerden geçirdim (float16). kaynak kelime
kontrolünü 0.8b için de tekrarladım, böylece üç boyut aynı 93 kelimede aynı ölçülerle karşılaştırılabiliyor.
tabloyu `src/stage4_boyut.py` üretiyor.

| model | yön | doğru (silmeden) | görünen ingilizce, logit / tuned | arındırılmış ingilizceyi silince p düşüşü | başlangıca oranı | kontrol dili | ilgisiz ingilizce |
|---|---|---|---|---|---|---|---|
| 0.8b | fr→tr | 15/93 | 0.075 / 0.067 | 0.013 [0.004, 0.025] | %18 | %2 | %1 |
| 0.8b | de→tr | 17/93 | 0.098 / 0.075 | 0.011 [0.001, 0.024] | %16 | %2 | %1 |
| 2b | fr→tr | 49/93 | 0.229 / 0.118 | 0.075 [0.052, 0.102] | %24 | %8 | %0 |
| 2b | de→tr | 51/93 | 0.251 / 0.107 | 0.069 [0.048, 0.091] | %22 | %-1 | %0 |
| 4b | fr→tr | 77/93 | 0.386 / 0.179 | 0.117 [0.082, 0.153] | %16 | %2 | %0 |
| 4b | de→tr | 80/93 | 0.375 / 0.143 | 0.092 [0.056, 0.129] | %12 | %0 | %0 |

"başlangıca oranı" p düşüşünün, silme yokken doğru cevaba verilen ortalama olasılığa bölümü.

- **büyüdükçe artan iki şey var.** çeviri becerisi (15 → 49 → 77 doğru) ve ara katlarda görünen ingilizce (logit lens
  tepe değeri 0.075 → 0.229 → 0.386). ekli kelimelerde tam doğru da aynı şekilde artıyor (2 → 14 → 40, 58 ifadede).
- **artmayan şey ingilizceye dayanma oranı.** arındırılmış ingilizceyi silmek üç boyutta da doğru cevabın olasılığının
  kabaca %12-24'ünü götürüyor. kontrol dili ve ilgisiz ingilizce kelime üç boyutta da %8'in altında, çoğu sıfıra yakın.
  0.8b'deki etki mutlak olarak küçüktü çünkü 0.8b doğru cevaba zaten düşük olasılık veriyor (ortalama 0.07).
- **en net sonuç 2b'de.** doğru cevap sayısı fr→tr'de 49'dan 38'e, de→tr'de 51'den 42'ye düşüyor. mcnemar'da
  13'e 2 (p 0.007) ve 10'a 1 (p 0.012). 4b'de aynı test sınırdaydı (p 0.07 ile 0.09), çünkü 4b çoğu kelimede o kadar emin
  ki olasılık düşse de en üstteki cevap değişmiyor. 0.8b'de ise az kelime doğru başladığı için değişecek bir şey yok.
- **benzemeyen kelimeler.** 2b'de fr→tr'nin benzemeyen 83 kelimesinde düşüş 0.081 [0.056, 0.110], doğru 46'dan 35'e.
  de→tr'nin benzemeyen 57 kelimesinde 0.086 [0.056, 0.116], doğru 30'dan 24'e. etki 4b'deki gibi kaynağa benzerlikten
  gelmiyor. 0.8b'de bu ayrımı yapacak güç yok.
- **kaynak kelimeyi silmek** 2b'de biraz daha etkili (0.047 ve 0.067), ama doğru sayısında anlamlı değil (p 0.29 ve 0.07).

bu tablo 6. bölümden sonraki hikâyemi değiştirdi. "küçük model ingilizce kullanmıyor, büyük model kullanıyor" demiştim.
üç boyut yan yana konunca daha doğru cümle şu oluyor. üç boyut da çevirirken ara katlarda ingilizceye hizalı yönü
kabaca aynı oranda kullanıyor. büyüdükçe değişen şey bu ingilizcenin ne kadar görünür olduğu ve modelin çeviride ne kadar
iyi olduğu. 0.8b'de "ingilizce yok" sonucu, testin gücünün yetmemesinden geliyordu.

bunun bir sınırı var. üç boyut ve iki dil çifti küçük bir örnek. "oran sabit" demek yerine "büyüklükle belirgin bir
artış ya da azalış yok" demek daha dürüst.

## ne çıkardım

1. qwen'in ara katları ingilizce kelimelere yakın, kumru'nunkiler türkçeye.
2. üç qwen boyutunda da (0.8b, 2b, 4b) o kavramın ingilizce kelimesine hizalı yönü ara katlardan silmek türkçe
   çeviriyi bozuyor. hedef ve kaynaktan arındırılmış haliyle bile doğru cevabın olasılığının %12-24'ü gidiyor.
   almancayı ya da ilgisiz bir ingilizce kelimeyi silmek bozmuyor. kaynak kelime kontrolü ve benzemeyen kelimeler
   (chien/dog) de bunu destekliyor. en net sonuç 2b'de, doğru cevap sayısı da anlamlı düşüyor (p 0.007 ve 0.012).
3. büyüdükçe değişen şey ingilizcenin görünürlüğü. logit lens'te ara katlardaki ingilizce 0.8b'de 0.075, 4b'de 0.386.
   0.8b'de tuned lens ingilizce ile türkçeyi başa baş gösteriyor, 4b'de ingilizce hedefin iki katı. 0.8b'de ilk
   müdahale testlerinin bir şey göstermemesi modelin çeviride zayıf olmasından (93 kelimenin 15'i) ve testin gücünün
   yetmemesinden geliyordu.
4. bu bir ingilizce cümle planı gibi görünmüyor. "evlerimizden" üretirken ara katlarda "house" var, "from" yok.
   ingilizce kavramın adı olarak duruyor, kelime sırası türkçenin.
5. kumru fransızcaya giderken ara katlarda ingilizce görünüyor ama silmek bir şey değiştirmiyor. köprü değil.
6. dil yönlerini "o dilin ortalaması eksi bütün dillerin ortalaması" diye bulmak tuzaklı. yönler birbirine karışıyor
   ve silme deneyleri yanıltıcı sonuç verebiliyor. kelimeye özel silme, olumlu kontrol (hedef kelimeyi silmek) ve
   ilgisiz kelime kontrolüyle çok daha temiz sonuç veriyor.

## açık sorular

- oranın büyüklükle değişmediğini üç boyuttan söylüyorum. 9b gibi daha büyük bir modelde aynı oran devam ediyor mu,
  bilmiyorum (t4'e sığmıyor).
- 4b tuned lens 800 adım eğitildi, kl hâlâ 2.4 ile 3.2 arasında gidip geliyor. daha uzun eğitim ingilizceyi biraz daha
  küçültebilir. silme sonucu mercekten bağımsız olduğu için ana bulguyu değiştirmez.
- silme sadece kelimenin ilk token'ının çıkış vektörünü kaldırıyor. kavramın ingilizce temsili bundan daha geniş olabilir,
  yani gerçek etki ölçtüğümden büyük olabilir.

## 8. çince kontrolü (11 kasım, `src/stage4_zh.py`)

soru: etki ingilizceye mi özgü, yoksa modelin en çok gördüğü dile mi? stage4_source ile aynı kelimeler, katlar, arındırma. kavramın çincesi (106 kelime, hiçbiri atlanmadı). ham sonuçlar `data/stage4_zh_*.json`.

| model, çift | hedef | en_dik2 | zh | zh_dik2 | kontrol_dik2 | en_dik2 − zh_dik2 |
|---|---|---|---|---|---|---|
| 4b fr-tr | %15 | %16 | %22 | %15 | %2 | +0.013 [−0.007, +0.033] |
| 4b de-tr | %15 | %12 | %17 | %10 | %0 | +0.012 [−0.009, +0.035] |
| 2b fr-tr | %59 | %24 | %35 | %24 | %8 | −0.002 [−0.020, +0.017] |
| 2b de-tr | %56 | %22 | %38 | %22 | %−1 | −0.002 [−0.020, +0.016] |

(göreli p düşüşü. son sütun mutlak p farkı ve %95 bootstrap aralığı.)

sonuç: çinceyi silmek ingilizceyi silmek kadar bozuyor. etki ingilizceye özgü değil. qwen kavramı çok gördüğü dillerin (ingilizce, çince) kelimelerine yakın tutuyor, almanca ve fransızca değil.

## 9. göreli etkinin güven aralığı (`src/istatistik.py`)

| model | çift | en_dik2 göreli | kontrol_dik2 göreli | fark |
|---|---|---|---|---|
| 0.8b | fr-tr | %18 [7, 29] | %2 [−5, 10] | 16 [5, 26] puan |
| 0.8b | de-tr | %16 [2, 28] | %2 [−5, 8] | 14 [2, 25] puan |
| 2b | fr-tr | %24 [17, 31] | %8 [3, 14] | 16 [10, 22] puan |
| 2b | de-tr | %22 [16, 28] | %−1 [−4, 3] | 22 [17, 28] puan |
| 4b | fr-tr | %16 [12, 21] | %2 [1, 4] | 14 [9, 19] puan |
| 4b | de-tr | %12 [7, 17] | %0 [−1, 1] | 12 [8, 16] puan |

aralıklar boyutlar arasında örtüşüyor. "oran boyutla değişmiyor" demek için yeterli değil ama belirgin bir artış ya da azalış da yok.
