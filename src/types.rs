//! Core value types: colours, pieces, squares, files, ranks.
//!
//! Everything here is a `u8` newtype with `const fn` constructors so tables can
//! be built at compile time. Squares are A1 = 0 .. H8 = 63, i.e. `sq = rank * 8 + file`.

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const COUNT: usize = 2;
    pub const ALL: [Color; 2] = [Color::White, Color::Black];

    #[inline(always)]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Direction a pawn of this colour advances, in ranks.
    #[inline(always)]
    pub const fn forward(self) -> i8 {
        match self {
            Color::White => 1,
            Color::Black => -1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const COUNT: usize = 6;
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn from_index(i: usize) -> PieceType {
        match i {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            5 => PieceType::King,
            _ => panic!("bad piece type index"),
        }
    }

    #[inline(always)]
    pub const fn char(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }
}

/// A coloured piece, packed as `color << 3 | piece_type`.
///
/// The gap at bit 3 keeps the two fields independently maskable and leaves
/// `Piece::NONE = 14` outside the range of any real piece, so a mailbox can be
/// a plain `[Piece; 64]` with no `Option` discriminant.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Piece(pub u8);

impl Piece {
    pub const NONE: Piece = Piece(14);

    #[inline(always)]
    pub const fn new(color: Color, pt: PieceType) -> Piece {
        Piece(((color as u8) << 3) | pt as u8)
    }

    #[inline(always)]
    pub const fn color(self) -> Color {
        if self.0 & 8 == 0 {
            Color::White
        } else {
            Color::Black
        }
    }

    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        PieceType::from_index((self.0 & 7) as usize)
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == Piece::NONE.0
    }

    #[inline(always)]
    pub const fn is_some(self) -> bool {
        self.0 != Piece::NONE.0
    }

    pub const fn char(self) -> char {
        if self.is_none() {
            return '.';
        }
        let c = self.piece_type().char();
        match self.color() {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    pub const fn from_char(c: char) -> Option<Piece> {
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let pt = match c.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _ => return None,
        };
        Some(Piece::new(color, pt))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Square(pub u8);

impl Square {
    pub const COUNT: usize = 64;

    #[inline(always)]
    pub const fn new(file: u8, rank: u8) -> Square {
        debug_assert!(file < 8 && rank < 8);
        Square(rank * 8 + file)
    }

    #[inline(always)]
    pub const fn from_index(i: usize) -> Square {
        debug_assert!(i < 64);
        Square(i as u8)
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    /// Mirror vertically (A1 <-> A8). Used for the black-perspective NNUE view
    /// and for building colour-symmetric tables from one colour's definition.
    #[inline(always)]
    pub const fn flip_rank(self) -> Square {
        Square(self.0 ^ 56)
    }

    #[inline(always)]
    pub const fn flip_file(self) -> Square {
        Square(self.0 ^ 7)
    }

    /// Offset by (file, rank), returning `None` if it would leave the board.
    /// The file check is what prevents the classic H-file-to-A-file wraparound.
    #[inline(always)]
    pub const fn offset(self, df: i8, dr: i8) -> Option<Square> {
        let f = self.file() as i8 + df;
        let r = self.rank() as i8 + dr;
        if f < 0 || f > 7 || r < 0 || r > 7 {
            None
        } else {
            Some(Square::new(f as u8, r as u8))
        }
    }

    pub fn from_uci(s: &str) -> Option<Square> {
        let b = s.as_bytes();
        if b.len() != 2 {
            return None;
        }
        let f = b[0].wrapping_sub(b'a');
        let r = b[1].wrapping_sub(b'1');
        if f < 8 && r < 8 {
            Some(Square::new(f, r))
        } else {
            None
        }
    }
}

impl std::fmt::Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}",
            (b'a' + self.file()) as char,
            (b'1' + self.rank()) as char
        )
    }
}

#[rustfmt::skip]
pub mod squares {
    use super::Square;
    macro_rules! def { ($($n:ident = $v:expr),* $(,)?) => { $(pub const $n: Square = Square($v);)* } }
    def!(
        A1= 0, B1= 1, C1= 2, D1= 3, E1= 4, F1= 5, G1= 6, H1= 7,
        A2= 8, B2= 9, C2=10, D2=11, E2=12, F2=13, G2=14, H2=15,
        A3=16, B3=17, C3=18, D3=19, E3=20, F3=21, G3=22, H3=23,
        A4=24, B4=25, C4=26, D4=27, E4=28, F4=29, G4=30, H4=31,
        A5=32, B5=33, C5=34, D5=35, E5=36, F5=37, G5=38, H5=39,
        A6=40, B6=41, C6=42, D6=43, E6=44, F6=45, G6=46, H6=47,
        A7=48, B7=49, C7=50, D7=51, E7=52, F7=53, G7=54, H7=55,
        A8=56, B8=57, C8=58, D8=59, E8=60, F8=61, G8=62, H8=63,
    );
}
