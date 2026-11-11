# aşama 5: türkçe sorular — deneyden önce yazılmış tahminler

yazıldığı an: 11 kasım 2026, hiçbir türkçe deney çalışmadan önce. bu dosya deneyler çalıştıktan sonra değiştirilmeyecek.
sonuçlar ayrı bir dosyaya (`results/stage5_notes.md`) yazılacak.

## kurulum

- model: qwen/qwen3.5-0.8b, greedy, 4096 yeni token, fp16, colab t4.
- sorular: math-500 test 200-299. türkçesi `bezir/MATH-500-multilingual` (Turkish), unique_id ile eşleşti, 100/100 cevap aynı.
  5 soruda çeviri düzeltildi (`data/stage5_tr_duzeltmeler.json`). 242, 263, 273'te sayılar türkçe biçimde (24.000, 6,5), dokunulmadı.
- talimat: ingilizce "Please reason step by step, and put your final answer within \boxed{}." /
  türkçe "Adım adım düşün ve son cevabını \boxed{} içine yaz."
- puanlama: math-verify. döngü: kelime bazlı (200 kelimede ≥180) ve token bazlı (256 token'da ≥%90 tekrar eden 8 token'lık parça).
- karşılaştırmalar soru başına eşli (mcnemar, kesin) ve %95 aralıkla.

## koşullar

1. en_off: ingilizce, düşünme kapalı (aynı oturumda yeniden)
2. tr_off: türkçe, düşünme kapalı
3. tr_on: türkçe, düşünme açık
4. tr_zorla / en_zorla: düşünme açık cevabı token bazlı döngü başında kesip `</think>` ekleme, 1024 token
5. geri_off: türkçe soruların ingilizceye geri çevrilmişi, düşünme kapalı (çeviri soruyu zorlaştırdı mı?)
6. tr_soru_en_talimat_off: türkçe soru + ingilizce talimat, düşünme kapalı

## tahminler

- tr_off doğru sayısı en_off'tan düşük olacak. tahminim 41'e karşı 28-35 arası (fark 6-13 puan).
- geri_off, en_off'a yakın olacak (±5). yani türkçedeki kaybın çoğu dilden, çeviriden değil.
- tr_off'ta sınıra dayanan cevap sayısı en_off'tan fazla olacak (33'e karşı 35-50).
- tr_on: 100 cevabın en az 90'ı sınıra dayanacak, en az 80'i birebir döngüye girecek. doğru 10 civarı ya da altı.
- düşünme açıkken model türkçe soruda da düşünce bloğunun çoğunu (>%70) ingilizce yazacak. son cevap kısmı da çoğunlukla ingilizce.
- tr_zorla doğru sayısı en_zorla'dan düşük ama tr_on'un 2 katından fazla olacak.
- tr_soru_en_talimat_off, tr_off'tan biraz iyi olacak (model ingilizce talimatla ingilizce düşünmeye kayar).

## yanılırsam ne demek

- tr_off ≈ en_off: 0.8b türkçe matematikte dil yüzünden kayıp yaşamıyor. 4. yazıyla birlikte: kavramı dilden bağımsız bir yerde tutuyor.
- geri_off << en_off: kaybın kaynağı çeviri (ya da ingilizce math'in ezberlenmiş olması), dil değil. bu durumda dil hakkında bir şey söyleyemem.
- tr_on'da döngü azsa: döngü ingilizceye özgü bir davranış, ilginç bir bulgu.
