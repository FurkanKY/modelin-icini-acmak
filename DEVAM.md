# devir notu (11 kasım 2026)

önceki oturum terminaldeki claude code'daydı. bu dosya yeni oturumun kaldığı yerden devam etmesi için.
her zaman türkçe cevap ver. adımları sade anlat. commit'lere claude atfı ekleme.

## proje

blog serisi "modelin içini açmak" (5 yazı). repo: github.com/FurkanKY/modelin-icini-acmak.
- aşama 1-3: qwen3.5-0.8b neden takılıyor, mit'nin astra deneyini elle tekrarlamak. notlar `results/stage2_3_notes.md`.
- aşama 4: türkçe konuşan model kafasında ingilizce mi düşünüyor (qwen 0.8b/2b/4b, kumru-2b).
  notlar `results/stage4_notes.md`, anlatım raporu `results/stage4_rapor.md` (+ pdf).
- stil kuralları `blog/STIL.md`. yayınlanmış ilk yazı `blog/01-...mdx`, sesi ondan öğren.

## bulgular kısaca (11 kasım akşamı itibarıyla)

1. takılma cevabın ilk 512 token'ında 15. katmandan %79.5 tahmin ediliyor. bu yön bir "bitirme düğmesi".
   rastgele yön kontrolü etkinin yöne özgü olduğunu gösterdi. doğruluk hiçbir müdahalede artmıyor.
   hafif itme yeni 100 soruda tekrarlanmadı (p 0.83), güçlü itme ve ters çevirme sağlam.
2. düşünme modu açıkken 100 cevabın 95'i birebir döngüye giriyor. döngüyü dışarıdan kesince (`</think>` ekleyip)
   doğru 10'dan 35'e çıkıyor. mit'nin kazancı büyük ihtimalle buradan. örnekleme, tekrar cezası, 16k sınır ve
   üç ayrı yön döngüyü kırmıyor. düşünme kapalı (41) yine zorla kesilmiş düşünmeden (35) iyi.
3. qwen kavramı ingilizceye yakın tutuyor ama çinceyi silmek de aynı kadar bozuyor. etki ingilizceye özel değil,
   çok görülen dillere yakınlık. almanca/fransızca bozmuyor. kumru kendi dilinde kalıyor.

## durum

- yazı taslakları 2, 3, 4 hazır ve güncel, hepsi `draft: true`. kullanıcı okuyacak.
- 7 commit push edilmedi (e7f6727 → 7474161). kullanıcı "gönder" demeden push yapma.
- seri 5 yazı diye duyuruluyor (`seriesTotal: 5`) ama 5. yazının konusu belirlenmedi.
- takip dışı: `results/stage4_rapor.pdf`, `blog/figures/01-*-narrow.svg`, bu dosya.

## sıradaki olası işler

- 5. yazı kararı: yazılacak mı, yoksa seri 4 yazıya mı inecek.
- 1. yazıya sonraki yazıların iddiayı yumuşattığına dair bir not.
- 4. yazıyı güçlendirmek için bir dil daha (ispanyolca ya da rusça) ile "çok görülen dil" testi.
- 3. yazıya bir şekil (10 → 35).
- kullanıcı onaylayınca push ve yazıların siteye taşınması.
