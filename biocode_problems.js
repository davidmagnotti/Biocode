const questions = [
  {
    "id": 1,
    "title": "DNA Transcription",
    "description": "Write a function that transcribes a DNA sequence into RNA. In transcription, DNA's thymine (T) is replaced by RNA's uracil (U), while other nucleotides remain the same. This is a fundamental process in molecular biology where genetic information flows from DNA to RNA.",
    "starterCode": "def transcribe_dna(dna_sequence):\n    return None",
    "example": "transcribe_dna(\"ATGCCCGTAAATT\") == \"AUGCCCGUAAAUU\"",
    "testCases": [
      {
        "input": "ATGCCCGTAAATT",
        "expected": "AUGCCCGUAAAUU",
        "code": "print(transcribe_dna(\"ATGCCCGTAAATT\"))"
      },
      {
        "input": "TTTTAAAACCCGGG",
        "expected": "UUUUAAAACCCGGG",
        "code": "print(transcribe_dna(\"TTTTAAAACCCGGG\"))"
      },
      {
        "input": "GATTACA",
        "expected": "GAUUACA",
        "code": "print(transcribe_dna(\"GATTACA\"))"
      }
    ]
  },
  {
    "id": 2,
    "title": "GC Content Calculator",
    "description": "Write a function that calculates the GC content percentage in a DNA sequence. GC content is the percentage of guanine (G) and cytosine (C) bases in a DNA sequence, which is important for understanding DNA stability and gene expression patterns.",
    "starterCode": "def calculate_gc_content(sequence):\n    return None",
    "example": "calculate_gc_content(\"GCGCGC\") == 100.0",
    "testCases": [
      {
        "input": "GCGCGC",
        "expected": "100.0",
        "code": "print(calculate_gc_content(\"GCGCGC\"))"
      },
      {
        "input": "AATTAATT",
        "expected": "0.0",
        "code": "print(calculate_gc_content(\"AATTAATT\"))"
      },
      {
        "input": "ATGCATGC",
        "expected": "50.0",
        "code": "print(calculate_gc_content(\"ATGCATGC\"))"
      }
    ]
  },
  {
    "id": 3,
    "title": "Reverse Complement",
    "description": "Write a function that returns the reverse complement of a DNA sequence. In DNA, A pairs with T, and C pairs with G. The reverse complement is important in DNA replication and PCR primer design. For example, the reverse complement of 'ATCG' is 'TAGC'.",
    "starterCode": "def reverse_complement(sequence):\n    return None",
    "example": "reverse_complement(\"ATCG\") == \"CGAT\"",
    "testCases": [
      {
        "input": "ATCG",
        "expected": "TAGC",
        "code": "print(reverse_complement(\"ATCG\"))"
      },
      {
        "input": "AAAAAA",
        "expected": "TTTTTT",
        "code": "print(reverse_complement(\"AAAAAA\"))"
      },
      {
        "input": "GATTACA",
        "expected": "CTAATGT",
        "code": "print(reverse_complement(\"GATTACA\"))"
      }
    ]
  },
  {
    "id": 4,
    "title": "K-mer Counter",
    "description": "Write a function that counts all k-mers (substrings of length k) in a DNA sequence. K-mers are crucial in genome assembly, read mapping, and sequence analysis. Return a dictionary with k-mers as keys and their frequencies as values.",
    "starterCode": "def count_kmers(sequence, k):\n    return None",
    "example": "count_kmers(\"ATATATATA\", 3) == {'ATA': 3, 'TAT': 2}",
    "testCases": [
      {
        "input": ["ATATATATA", 3],
        "expected": "{'ATA': 3, 'TAT': 2}",
        "code": "print(count_kmers(\"ATATATATA\", 3))"
      },
      {
        "input": ["GATTACA", 2],
        "expected": "{'GA': 1, 'AT': 1, 'TT': 1, 'TA': 1, 'AC': 1, 'CA': 1}",
        "code": "print(count_kmers(\"GATTACA\", 2))"
      },
      {
        "input": ["AAAA", 2],
        "expected": "{'AA': 3}",
        "code": "print(count_kmers(\"AAAA\", 2))"
      }
    ]
  }
];
