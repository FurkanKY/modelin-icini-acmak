# Aşama 4 anlatım raporu: Türkçe konuşan bir model kafasında İngilizce mi düşünüyor?

Bu rapor blog yazısını yazarken elinin altında dursun diye hazırlandı. Her bölümde üç katman var:

- **Çocuğa anlatır gibi:** kavramı hiç bilmeyen birine ilk cümlede ne denir.
- **Teknik olarak:** okurun içinden "tam olarak ne yaptın" diye soran kısmına cevap.
- **Blogda:** bu kısmı yazıda nasıl tanıtabileceğin ve kendi cümleni nereye koyacağın.

`[senin cümlen: ...]` yazan yerler bilerek boş bırakıldı. Orada senin gözlemin, şaşkınlığın ya da yorumun olmalı.
Rakamların hepsi `data/` klasöründeki dosyalardan geliyor, hangi dosyadan geldiği yanında yazıyor.
Ayrıntılı ham notlar `results/stage4_notes.md` dosyasında.

---

## 1. Tek paragrafta ne bulduk

Dört modele Fransızcadan ve Almancadan Türkçeye kelime çevirttik ve modelin içine katman katman baktık.
Türkçe eğitilmiş model (Kumru-2B) içeride Türkçe düşünüyor, Fransızcaya giderken İngilizce görünse bile onu
kullanmıyor. Çok dilli Qwen3.5'in üç boyutu (0.8B, 2B, 4B) ise İngilizceden geçiyor. Çeviri sırasında modelin içinden
o kavramın İngilizce kelimesini silince Türkçe cevabın olasılığı üç boyutta da kabaca %12-24 düşüyor. Aynı kavramın
Almancasını ya da alakasız bir İngilizce kelimeyi silmek bir şey yapmıyor. Büyüdükçe değişen şey İngilizceye ne kadar
dayandığı değil, İngilizcenin ne kadar görünür olduğu. Küçük modelde görünen İngilizcenin bir kısmı ölçüm aracının
yanılsaması, büyük modelde değil. Model "evlerimizden" derken
içeride "house" var ama "from" yok. Yani İngilizce bir cümle kurmuyor, kavramın adını İngilizce tutuyor.

---

## 2. Yazı için önerilen akış

Bu sıra okuru merakla baştan sona götürür. Her adımda bir soru, bir deney, bir sürpriz var.

1. **Soru:** "Türkçe konuşan bir yapay zekâ aslında kafasında İngilizce mi düşünüyor?" Herkesin aklına bir kere
   gelmiş bir soru. `[senin cümlen: bu soru sana nereden geldi?]`
2. **Önceki çalışma:** Wendler ve arkadaşları 2024'te Llama-2 için "evet, İngilizceden geçiyor" dedi. Ama Türkçe
   ile ve Türkçe eğitilmiş bir modelle bakan olmadı.
3. **Araç:** Logit lens ile modelin içine bakmak. İlk grafik (`04-ara-dil.svg`) burada.
4. **İlk sonuç:** Qwen'de İngilizce görünüyor, Kumru'da görünmüyor. "Tamam, cevap bu" diye düşünürsün.
5. **İlk sürpriz:** Daha iyi bir mercek (tuned lens) ile bakınca küçük Qwen'deki İngilizce neredeyse kayboluyor.
   Görünen şeyin bir kısmı merceğin kendisinden geliyormuş.
6. **İkinci sürpriz:** Silme deneyleri. İlk denemede "İngilizceyi silince çeviri bozuldu" sonucu çıktı ve yanlış
   çıktı. Dil yönleri birbirine karışıyordu. `[senin cümlen: yanlış sonuca az kalsın inandığın an]`
7. **Doğru test:** Kelimeye özel silme ve kontroller. Kumru'da etki yok, küçük Qwen'de belli belirsiz.
8. **Büyük model:** Qwen3.5-4B'de etki var ve kontrollerden geçiyor.
9. **Üçüncü sürpriz:** Aradaki 2B'yi de ekleyince hikâye değişiyor. Etkinin başlangıca oranı üç boyutta da benzer.
   Küçük modelde "etki yok" sanılan şey testin gücünün yetmemesiymiş. `[senin cümlen: hikâyenin değiştiği an]`
10. **Son şüphe ve çürütülmesi:** "Belki kaynak kelimeyi siliyorsun?" Benzemeyen kelimelerde (chien/dog) de aynı etki.
11. **"evlerimizden" ve "from":** İngilizce cümle değil, kavramın adı.
12. **Sınırlar ve açık sorular.** Dürüst kapanış.

---

## 3. Kavramlar

Yazıda tanıtman gereken kavramlar aşağıda, okura çıkma sırasıyla. Hepsini tanıtmak zorunda değilsin. Yıldızlı
olanlar hikâye için şart.

### Token (parça) ★

- **Çocuğa anlatır gibi:** Model kelimeleri bütün halinde okumaz, yapboz parçaları gibi küçük parçalara böler.
  "evlerimizden" onun için "ev" + "lerimiz" + "den" gibi birkaç parça olabilir.
- **Teknik olarak:** Tokenizer metni sözlükteki alt-kelime birimlerine ayırır. Her token bir sayıya, her sayı bir
  vektöre karşılık gelir. Türkçe gibi eklemeli dillerde bir kelime çoğu zaman birkaç token eder.
- **Blogda:** Bir kelimenin nasıl parçalandığını gösteren tek bir örnek yeter. Neden önemli olduğu 10. adımda
  ortaya çıkıyor.

### Katman ve gizli durum ★

- **Çocuğa anlatır gibi:** Model bir fabrika bandı gibi. Kelime bandın başından giriyor, her istasyonda biraz
  daha işleniyor, sonunda cevap çıkıyor. Biz bandı ortadan durdurup "şu an elindeki ne?" diye bakıyoruz.
- **Teknik olarak:** Transformer katmanları, her token için bir vektörü (gizli durum, residual stream) adım
  adım günceller. Qwen3.5-0.8B'de 24, Qwen3.5-4B'de 32, Kumru-2B'de 18 katman var.
- **Blogda:** "Ara katman" sözünü ilk kez kullandığın yerde bu benzetmeyi ver.

### Logit lens ★

- **Çocuğa anlatır gibi:** Bandın ortasındaki yarım işi alıp "bu haliyle cevap verseydin ne derdin?" diye
  soruyoruz. Model yarım işi zorla sonuca çeviriyor ve biz o tahmini okuyoruz.
- **Teknik olarak:** Ara katmandaki gizli durumu doğrudan modelin son normalizasyon katmanından ve çıkış
  matrisinden geçirip kelimeler üzerindeki olasılık dağılımına bakıyoruz (nostalgebraist 2020).
- **Blogda:** İlk grafiğin altına "bu yöntem hızlı ama kaba" notunu düş. 5. adımdaki sürpriz buna dayanıyor.

### Tuned lens ★

- **Çocuğa anlatır gibi:** Logit lens, yabancı dilde konuşan birini sözlüksüz anlamaya çalışmak gibi. Tuned
  lens ise her istasyon için küçük bir tercüman eğitiyor. Tercüman o istasyonun "dilini" son istasyonun diline
  çeviriyor.
- **Teknik olarak:** Her katman için bir afin dönüşüm (matris + sapma), ara katmanın tahmini son katmanın
  tahminine en çok benzeyecek şekilde (KL uzaklığı) eğitiliyor. Model değişmiyor, sadece okuma aracı değişiyor
  (Belrose ve ark. 2023). Dört dilde eşit miktarda Vikipedi metniyle eğittik.
- **Blogda:** "Ölçüm aletini de sorguladım" kısmının kahramanı bu. Okur için önemli mesaj şu: logit lens, modelin
  çıkışı hangi dile alışkınsa o dili abartıyor. Qwen'de İngilizceyi, Kumru'da Türkçeyi.

### Silme (ablation) ★

- **Çocuğa anlatır gibi:** Bir yemeğin tadında tuzun payını anlamak istiyorsan tuzsuz pişirip tadına bakarsın.
  Biz de modelin içinden bir şeyi çıkarıp cevabın bozulup bozulmadığına bakıyoruz.
- **Teknik olarak:** Belirli katmanlarda gizli durumdan belirli bir yönün (vektörün) payını çıkarıyoruz
  (izdüşümü siliyoruz) ve son katmanda doğru cevaba verilen olasılığa bakıyoruz.
- **Blogda:** "Görmek" ile "kullanmak" arasındaki farkı buradan anlatabilirsin. Logit lens görür, silme
  kullanılıp kullanılmadığını test eder.

### Kelimenin çıkış vektörü

- **Çocuğa anlatır gibi:** Modelin sonunda her kelimenin bir "çağrı düğmesi" var. Model bir kelimeyi söylemek
  istediğinde o düğmeye doğru itiyor. Biz ara katmanlardan "house" düğmesine giden yolu kapatıyoruz.
- **Teknik olarak:** Çıkış (unembedding) matrisinde kelimenin ilk token'ına ait satır, son normalizasyonun
  ağırlıklarıyla çarpılmış haliyle. Ara katmandaki gizli durumdan bu yönü siliyoruz.
- **Blogda:** Teknik okur için bir cümleyle yeter. Asıl anlatılacak olan kontroller.

### Kontroller (olumlu, olumsuz) ★

- **Çocuğa anlatır gibi:** Termometreye güvenmek için önce kaynar suya sokup 100 gösteriyor mu diye bakarsın
  (olumlu kontrol). Sonra oda sıcaklığında yanlışlıkla 100 göstermediğinden emin olursun (olumsuz kontrol).
- **Teknik olarak:**
  - **Olumlu kontrol:** Türkçe hedef kelimenin kendi vektörünü silmek. Bu cevabı bozmalı, bozuyor da. Kumru'da
    doğru cevap 0'a iniyor.
  - **Olumsuz kontroller:** Aynı kavramın Almancasını silmek ve alakasız bir kavramın İngilizcesini silmek. Bunlar
    bozmamalı.
  - **Arındırma:** İngilizce vektörden hedef ve kaynak kelimeye ait payı çıkarıp kalanını silmek. İngilizce
    kelime hedefe biraz benziyorsa yanlışlıkla hedefi silmiş olmayalım diye.
- **Blogda:** Yazının güvenilirliğini taşıyan kısım bu. Okura "neden buna inanayım?" sorusunun cevabını veriyor.

### Benzer kelimeler (cognate)

- **Çocuğa anlatır gibi:** Almanca "Haus" ile İngilizce "house" kardeş gibi, birbirine benziyor. Fransızca
  "chien" ile İngilizce "dog" ise hiç benzemiyor.
- **Teknik olarak:** Kaynak kelime ile İngilizcesinin yazılış benzerliğini ölçüp (difflib oranı, 0.55 eşiği)
  kelimeleri ikiye ayırdık. Etki sadece benzer kelimelerde olsaydı, sildiğimiz şey İngilizce değil kaynak kelime
  olurdu.
- **Blogda:** Son şüphenin çürütüldüğü an. Kısa ve vurucu anlatılabilir.

### Güven aralığı ve bootstrap

- **Çocuğa anlatır gibi:** 93 kelimelik sınavı torbaya koyup rastgele tekrar tekrar çekiyoruz ve sonucun ne kadar
  oynadığına bakıyoruz. Hep aynı yere düşüyorsa sonuç şans değil.
- **Teknik olarak:** Kelimeleri yerine koyarak 10.000 kez yeniden örnekleyip ortalama farkın %95 aralığını
  bulduk. Doğru/yanlış sayıları için McNemar testi kullandık. Bu test sadece iki koşulda farklı sonuç veren
  kelimelere bakar.
- **Blogda:** Tek bir cümle ve aralıkları tabloya koymak yeterli.

---

## 4. Deneyler, sırasıyla

### 4.1 Logit lens ile ilk bakış

- **Soru:** Model Fransızcadan Türkçeye çevirirken ara katmanlarda hangi dilin kelimesi öne çıkıyor?
- **Ne yaptık:** Modele 4 örnek çeviri gösterip beşinciyi istedik (`Français: "chien" - Türkçe: "`). Görevde hiç
  İngilizce yok. 106 kelime, her katmanda İngilizce, kaynak ve hedef cevabın olasılığına baktık.
- **Sonuç** (`data/stage4_*_fr-tr.json`, ara katlardaki en yüksek ortalama):

| model | İngilizce | Türkçe (hedef) | İngilizce Türkçeyi geçiyor |
|---|---|---|---|
| Qwen3.5-0.8B | 0.075 | 0.035 | 71/93 kelime |
| Qwen3.5-2B | 0.229 | 0.044 | 80/93 kelime |
| Qwen3.5-4B | 0.386 | 0.022 | 92/93 kelime |
| Kumru-2B | 0.002 | 0.304 | 10/92 kelime |

- **Çocuğa anlatır gibi:** Qwen'in fabrika bandının ortasında "dog" yazan bir kâğıt duruyor, sonunda "köpek"
  çıkıyor. Kumru'nun bandında baştan beri "köpek" var.
- **Dikkat:** Bu sadece okuma. Model bu İngilizceyi kullanıyor mu, bilmiyoruz.
- **Grafik:** `blog/figures/04-ara-dil.svg` (dört panel, logit lens).
- `[senin cümlen: 4B'nin grafiğini ilk gördüğünde ne düşündün?]`

### 4.2 Tuned lens: mercek bizi kandırıyor mu?

- **Soru:** Görünen İngilizce gerçek mi, yoksa logit lens mi abartıyor?
- **Sonuç** (`data/stage4_tuned_*.json`, fr→tr):

| model | mercek | İngilizce | Türkçe | İngilizce önde |
|---|---|---|---|---|
| Qwen3.5-0.8B | logit | 0.075 | 0.035 | 71/93 |
| Qwen3.5-0.8B | tuned | 0.067 | 0.075 | 46/93 |
| Qwen3.5-2B | logit | 0.229 | 0.044 | 80/93 |
| Qwen3.5-2B | tuned | 0.118 | 0.104 | 49/93 |
| Qwen3.5-4B | logit | 0.386 | 0.022 | 92/93 |
| Qwen3.5-4B | tuned | 0.179 | 0.092 | 64/93 |
| Kumru-2B (tr→fr) | logit | 0.092 | 0.656 (kaynak) | |
| Kumru-2B (tr→fr) | tuned | 0.062 | 0.072 (kaynak) | |

- **Çocuğa anlatır gibi:** Küçük Qwen'de tercüman gelince "dog" kâğıdı küçülüyor, "köpek" kâğıdıyla aynı boya
  geliyor. Orta boy Qwen'de "köpek"ten biraz büyük kalıyor. Büyük Qwen'de küçülüyor ama hâlâ "köpek"in iki katı.
- **Ne anlama geliyor:** Logit lens modelin çıkışının alışkın olduğu dili abartıyor. Küçük Qwen'deki İngilizcenin
  çoğu bu abartı. Büyük Qwen'deki İngilizce ise tercümandan sonra da kalıyor.
- **Dikkat:** Tercümanlar kısa eğitildi (0.8B için 1500, 2B ve 4B için 800 adım). Daha uzun eğitim sayıları biraz
  oynatabilir.

### 4.3 İlk silme denemesi ve tuzak

- **Ne yaptık:** Her dil için bir "yön" bulduk (o dildeki kelimelerin ortalaması eksi bütün dillerin ortalaması) ve
  İngilizce yönünü ara katlardan sildik.
- **Ne oldu:** Küçük Qwen'in Türkçe çevirisi çöktü (6 doğrudan 0'a). Heyecan verici görünüyordu.
- **Neden yanlıştı:** İngilizce yönü ile Türkçe yönü birbirinin neredeyse tersiydi (kosinüs -0.60). İngilizceyi
  silerken Türkçeyi de siliyorduk. İngilizce yönünden öteki dillerin payını çıkarınca etki kayboldu (6 doğrudan 5'e).
  Kumru'da da aynı tuzak çıktı (tr→fr'de 11 doğrudan 5'e, arındırınca 11).
- **Çocuğa anlatır gibi:** Tuzu çıkarayım derken yanlışlıkla şekeri de çıkarmışız. Tat bozuldu ama tuz yüzünden
  değil.
- **Blogda:** Bu kısım yazının en insani anı olabilir. `[senin cümlen: yanlış sonucu fark ettiğin an]`

### 4.4 Kelimeye özel silme: Kumru ve küçük Qwen

- **Ne yaptık:** Her kelime için sadece o kavramın İngilizce kelimesinin çıkış vektörünü sildik. Yanına
  kontrolleri koyduk.
- **Sonuç** (`data/stage4_bridge_*_kelime.json`, doğru başlayan cevap sayısı):

| model, yön | hiçbiri | İngilizce (arındırılmış) | Almanca (arındırılmış) | Türkçe hedef |
|---|---|---|---|---|
| Kumru, tr→fr | 18/92 | 18 | 16 | 0 |
| Kumru, fr→tr | 38/92 | 40 | 38 | 1 |
| Qwen 0.8B, fr→tr | 15/93 | 14 | 17 | 6 |
| Qwen 0.8B, tr→fr | 45/93 | 40 | 42 | 7 |

- **Ne anlama geliyor:** Yöntem çalışıyor, hedefi silince cevap çöküyor. Kumru'da İngilizceyi silmek hiçbir şey
  değiştirmiyor. Kumru Fransızcaya giderken ara katlarda İngilizce görünse de onu köprü olarak kullanmıyor.
  Küçük Qwen'de etki burada ölçülemeyecek kadar küçük. Model çeviride zaten zayıf (93'te 15), küçük bir etkiyi
  yakalamak zor. 4.6b'de bu sonucun neden yanıltıcı olduğu var.

### 4.5 Asıl bulgu: Qwen3.5-4B

- **Soru:** Büyük model içeride gördüğümüz İngilizceyi kullanıyor mu?
- **Sonuç** (`data/stage4_source_Qwen3.5-4B.json`, fr→tr, 93 kelime):

| silinen | doğru | doğru cevabın olasılığı ne kadar düştü [%95 aralık] |
|---|---|---|
| hiçbiri | 77/93 | |
| Türkçe hedef (olumlu kontrol) | 65 | 0.110 [0.081, 0.141] |
| Fransızca kaynak kelime | 75 | 0.030 [0.014, 0.049] |
| İngilizce | 64 | 0.230 [0.185, 0.277] |
| İngilizce, hedeften ve kaynaktan arındırılmış | 70 | 0.117 [0.082, 0.153] |
| Almanca, arındırılmış | 76 | 0.017 [0.007, 0.028] |
| alakasız bir İngilizce kelime, arındırılmış | 76 | -0.001 [-0.003, 0.001] |

  de→tr'de aynı tablo: arındırılmış İngilizce 0.092 [0.056, 0.129], Fransızca kontrol 0.000, alakasız İngilizce -0.001.

- **Çocuğa anlatır gibi:** Büyük Qwen "köpek" demeden önce kafasında "dog" düşünüyor. Elinden "dog"u alınca
  "köpek"i bulmakta zorlanıyor. Elinden "Hund"u (Almancası) ya da alakasız bir "apple"ı alınca hiç zorlanmıyor.
- **İlginç ayrıntı:** İngilizceyi silmek, Türkçe hedefin kendisini silmekten daha fazla zarar veriyor. Çünkü
  silme yaptığımız katmanlarda (16-27) Türkçe henüz ortada yok, 30. katmandan sonra geliyor. O katmanlarda kavramı
  taşıyan şey İngilizceye hizalı yön.
- **Dikkat:** Olasılıktaki düşüş çok net. En üstteki cevabın değiştiği kelime sayısı daha az (77'den 70'e) ve
  McNemar testinde sınırda (p 0.04 ile 0.15 arası). Blogda "olasılık" üzerinden anlatmak daha dürüst.
- `[senin cümlen: bu tabloyu gördüğün an]`

### 4.6 Son şüphe: kaynak kelime mi siliniyor?

- **Şüphe:** Almanca "Haus" ile İngilizce "house" birbirine benziyor. İngilizceyi silerken kaynak kelimenin
  bilgisini silmiş olabiliriz.
- **İki cevap** (`data/stage4_source_Qwen3.5-4B_benzerlik.json`):
  1. Kaynak kelimeyi doğrudan silmek çok az zarar veriyor (0.03). Model kaynak kelimeyi bu yönde pek tutmuyor.
  2. Kelimeleri ikiye ayırdık. Benzemeyen kelimelerde etki tam gücüyle var:

| yön | grup | örnek | arındırılmış İngilizceyi silince düşüş |
|---|---|---|---|
| fr→tr | benzemeyen (83) | livre/book, chien/dog | 0.112 [0.074, 0.149] |
| fr→tr | benzer (10) | nuit/night, feu/fire | 0.165 [0.074, 0.270] |
| de→tr | benzemeyen (57) | Buch/book, Hund/dog | 0.120 [0.076, 0.171] |
| de→tr | benzer (36) | Haus/house, Herz/heart | 0.047 [-0.007, 0.100] |

- **Çocuğa anlatır gibi:** "dog" kâğıdı "chien" kâğıdına hiç benzemiyor. Yine de "dog"u alınca model bocalıyor.
  Demek ki bocalamanın sebebi kâğıtların birbirine benzemesi olamaz.

### 4.6b Aradaki boyut: Qwen3.5-2B ve üç boy yan yana

- **Soru:** 0.8B ile 4B arasındaki fark büyüklükten mi geliyor? Küçük model İngilizce kullanmıyor mu, yoksa biz mi
  ölçemiyoruz?
- **Ne yaptık:** 2B'yi aynı testlerden geçirdik. 4.5'teki kaynak kelime kontrolünü 0.8B için de tekrarladık. Üç boyut
  aynı 93 kelimede, aynı ölçülerle (`src/stage4_boyut.py`).
- **Sonuç** (fr→tr, `data/stage4_source_*.json`, `data/stage4_tuned_*.json`):

| model | doğru (silmeden) | görünen İngilizce (logit) | arındırılmış İngilizceyi silince olasılığın kaybolan payı |
|---|---|---|---|
| Qwen3.5-0.8B | 15/93 | 0.075 | %18 (kontrol dili %2) |
| Qwen3.5-2B | 49/93 | 0.229 | %24 (kontrol dili %8) |
| Qwen3.5-4B | 77/93 | 0.386 | %16 (kontrol dili %2) |

  de→tr'de aynı sıra: %16, %22, %12. Alakasız İngilizce kelime üç boyutta da %1'in altında.

- **Çocuğa anlatır gibi:** Üç kardeş düşün. Küçüğü "köpek"i zar zor biliyor, büyüğü çok iyi biliyor. Üçünün de
  elinden "dog" kâğıdını alınca bildiklerinin yaklaşık beşte biri gidiyor. Küçüğün kaybı küçük görünüyordu çünkü
  zaten az şey biliyordu.
- **En net sonuç 2B'de:** Doğru cevap sayısı 49'dan 38'e (McNemar 13'e 2, p 0.007), de→tr'de 51'den 42'ye (10'a 1,
  p 0.012) düşüyor. 4B o kadar emin ki olasılık düşse de en üstteki cevap çoğu kelimede değişmiyor. 0.8B'de ise
  değişecek kadar doğru cevap yok. Kaynağa benzemeyen kelimelerde de (chien/dog) 2B'de etki tam gücüyle var.
- **Ne anlama geliyor:** Büyüdükçe artan şey çeviri becerisi ve İngilizcenin görünürlüğü. İngilizceye dayanma oranında
  belirgin bir artış ya da azalış yok.
- **Dikkat:** Üç boyut ve iki dil çifti küçük bir örnek. "Oran sabit" yerine "boyutla belirgin bir değişim yok" demek
  daha dürüst. Kontrol dilinin 2B fr→tr'de %8'e çıkması da göz ardı edilmemeli.
- `[senin cümlen: hikâyenin "küçük kullanmıyor, büyük kullanıyor"dan buna dönmesi]`

### 4.7 "evlerimizden": İngilizce cümle mi, İngilizce kavram mı?

- **Soru:** Model İngilizce bir cümle kurup ("from our houses") sonra mı çeviriyor?
- **Ne yaptık:** "de nos maisons" gibi 60 ifade verdik. Türkçe cevap tek kelime ("evlerimizden") ve kökle başlıyor.
  İngilizcesi edatla ("from") başlıyor. Ara katlarda Türkçe köke, İngilizce edata ve İngilizce isme baktık.
- **Sonuç** (`data/stage4_suffix_Qwen3.5-4B.json`, tuned lens). 4B 58 ifadenin 40'ını tam doğru yazdı (2B 14, 0.8B 2).

| katman | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
|---|---|---|---|---|---|---|---|
| "house" | 0.041 | 0.087 | 0.150 | 0.080 | 0.011 | 0.001 | 0.000 |
| "ev" | 0.010 | 0.012 | 0.022 | 0.056 | 0.235 | 0.439 | 0.715 |
| "from" | 0.001 | 0.001 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 |

- **Çocuğa anlatır gibi:** Model kafasında "house" resmini tutuyor ama "from our houses" cümlesini kurmuyor.
  Cümleyi baştan Türkçe kuruyor, sadece "ev"in yerinde bir süre "house" duruyor.
- **Dikkat:** Bu okuma, silme değil. Ayrıca "from" gibi bir bağlaç kelimesinin ara katlarda görünmesini
  beklemek ne kadar adil, tartışılabilir. Güçlü bir iddia olarak değil, ipucu olarak sun.

---

## 5. Neyi söyleyebiliriz, neyi söyleyemeyiz

**Söyleyebiliriz:**

- Qwen3.5 (0.8B, 2B, 4B) Fransızcadan ve Almancadan Türkçeye çevirirken ara katmanlarda kavramı İngilizce kelimeye
  hizalı bir yönde tutuyor ve bu yön çeviri için gerekli. En net kanıt 2B'de, en büyük mutlak etki 4B'de.
- İngilizceye dayanma oranı boyutla belirgin şekilde değişmiyor. Büyüdükçe artan şey İngilizcenin görünürlüğü.
- Bu etki aynı kavramın Almancasında yok, alakasız İngilizce kelimelerde yok ve kaynağa benzemeyen kelimelerde de var.
- Türkçe eğitilmiş Kumru'da böyle bir etki yok.
- Logit lens, modelin çıkışının alışkın olduğu dili abartıyor. Küçük Qwen'deki İngilizcenin çoğu bundan geliyor.
- Dil yönlerini ortalamalarla bulup silmek yanıltıcı sonuç verebiliyor.

**Söyleyemeyiz:**

- "Model İngilizce cümle kurup çeviriyor." Ekli kelimelerde "from" hiç görünmedi.
- "Kavram İngilizcedir." Testimiz, kavram dilden bağımsız olup İngilizceye en yakın duruyor olsa da aynı sonucu
  verirdi. Almancanın işe yaramaması İngilizceye özgü bir şey olduğunu düşündürüyor ama tam ayıramıyoruz.
- "Büyüdükçe modeller daha çok İngilizce düşünüyor." Görünür İngilizce artıyor, dayanma oranı artmıyor. Bu ikisi
  karıştırılmamalı.
- "Bütün boyutlarda oran aynı." Üç boyut ve iki dil çifti var. 9B ve üstüne bakamadık (T4'e sığmıyor).

---

## 6. Çocuğa anlatır gibi: bütün hikâye

> Bilgisayardaki konuşan programlar, kelimeleri bir fabrika bandından geçirerek cevap üretiyor. Biz bandı ortadan
> durdurup "şu an elinde ne var?" diye baktık.
>
> Türkçe öğrenmiş bir program (Kumru) "köpek" derken bandın ortasında da hep "köpek" düşünüyordu. Birçok dil
> öğrenmiş programlarda (Qwen) bandın ortasında "dog" yazan bir kâğıt gördük.
>
> Ama sadece görmek yetmez. Büyük Qwen'in "dog" kâğıdını aldık, "köpek"i bulmakta zorlandı. Almanca "Hund"
> kâğıdını ya da alakasız bir "elma" kâğıdını alınca hiç zorlanmadı. Küçük Qwen'in kâğıdını alınca ilk başta bir
> şey olmadı sandık. Ortanca kardeşi de ekleyince anladık ki üçü de aşağı yukarı aynı oranda zorlanıyor. Küçüğü
> zaten az şey bildiği için kaybı gözümüze küçük görünmüş.
>
> Yani Qwen gerçekten "dog" diye düşünüp "köpek" diyor. Ama "evlerimizden" derken kafasında İngilizce bir
> cümle yok. Sadece "ev"in yerinde bir süre "house" duruyor.

---

## 7. Grafikler

- `blog/figures/04-ara-dil.svg` ve `-narrow` sürümü: dört model yan yana (Qwen 0.8B, 2B, 4B, Kumru), logit lens. İngilizce turuncu, Türkçe gri.
  Altına mutlaka "logit lens İngilizceyi abartıyor, tuned lens ile bakınca küçük modeldeki tepe büyük ölçüde
  kayboluyor" notu.
- İngilizce sürüm: `04-latent-language.svg`.
- Henüz çizilmemiş ama yazıyı en çok güçlendirecek grafik, 4.5'teki silme tablosunun yatay çubuk hali. Her
  satır silinen şey, çubuk olasılık düşüşü, aralıklar ince çizgi. İstersen bunu çizerim.

---

## 8. Önceki çalışmalar (yazıda anmak için)

- **Wendler ve ark. 2024, "Do Llamas Work in English?"** Llama-2'de logit lens ile ara katmanlarda İngilizceyi
  gösterdi. Bizim çeviri kurgumuz onlarınki.
- **nostalgebraist 2020:** logit lens fikri.
- **Belrose ve ark. 2023:** tuned lens.
- **Dumas ve ark. 2025:** çeviri sırasında ara katmanları başka bir çalıştırmaya yapıştırma (patching).
- **Zhong ve ark. 2025:** İngilizce merkezli olmayan modellerde birden fazla iç dil (Japonca).
- **Bayazıt ve ark. 2026:** logit lens'in İngilizceyi abarttığını, farklı ölçüm araçlarının anlaşamadığını gösterdi.

Bizim katkımız şunlar:
- Türkçe ile ve sıfırdan Türkçe eğitilmiş bir modelle (Kumru) karşılaştırma.
- Eklemeli kelimelere bakış ("evlerimizden").
- Kontrollü, kelimeye özel bir silme testiyle büyük modelde İngilizcenin kullanıldığını göstermek.

---

## 9. Blog için hazır parçalar

**Başlık seçenekleri:**

- Türkçe konuşan yapay zekâ kafasında İngilizce mi düşünüyor?
- "köpek" demeden önce "dog" diyen model
- Modelin içinde İngilizce gördüm, sonra neredeyse hepsinin yanılsama olduğunu öğrendim
- Model büyüdükçe İngilizce daha çok görünüyor ama daha çok kullanılmıyor

**Açılış için cümle iskeletleri:**

- "Bir yapay zekâya Türkçe soru sorduğumda kafasında önce İngilizceye çevirip çevirmediğini hep merak etmiştim.
  `[senin cümlen]`"
- "Bu soruya cevap veren bir makale vardı ama Türkçeye bakmamıştı. Ben de baktım. `[senin cümlen]`"

**Dönüm noktalarında kullanabileceğin cümleler:**

- "Grafik çok güzeldi. Fazla güzeldi." (tuned lens öncesi)
- "İlk silme deneyinde sonuç tam istediğim gibi çıktı. Sonra neden çıktığını anladım." (tuzak)
- "Kontrolleri ekleyince küçük modeldeki etki neredeyse kayboldu. Büyük modelde ise yerinde durdu." (asıl bulgu)
- "Arada bir boyut daha deneyince küçük modelin de aynı şeyi yaptığını, sadece benim göremediğimi anladım." (2B)
- "chien ile dog arasında hiçbir benzerlik yok. Yine de dog'u silince model köpeği bulamadı." (son şüphe)

**Kapanış için:**

- Açık sorular: daha büyük modeller (9B), daha uzun tuned lens eğitimi, kelimenin sadece ilk parçasını silmenin sınırı.
- `[senin cümlen: bu deneyden sonra yapay zekâya Türkçe yazarken ne düşünüyorsun?]`

---

## 10. Kod ve veri haritası

| ne | kod | veri |
|---|---|---|
| logit lens çeviri | `src/stage4_latent.py` | `data/stage4_{model}_{yön}.json` |
| dil yönü silme (ilk deneme) | `src/stage4_causal.py` | `data/stage4_causal_*.json` |
| yama (patching) | `src/stage4_patch.py` | `data/stage4_patch_*.json` |
| kavram enjeksiyonu | `src/stage4_inject.py` | `data/stage4_inject_*.json` |
| tuned lens | `src/stage4_tunedlens.py` | `data/stage4_tuned_*.json` |
| ekli kelimeler | `src/stage4_suffix.py` | `data/stage4_suffix_*.json` |
| dik yönler, kelimeye özel silme | `src/stage4_bridge.py` | `data/stage4_bridge_*.json` |
| kaynak kelime kontrolü | `src/stage4_source.py` | `data/stage4_source_{model}.json` |
| benzer/benzemeyen ayrımı | `src/stage4_source_analiz.py` | `data/stage4_source_{model}_benzerlik.json` |
| üç boyut yan yana | `src/stage4_boyut.py` | yukarıdaki dosyalardan okuyor |
| grafik | `src/charts.py` (depoda değil) | `blog/figures/04-*.svg` |
